import collections
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np

CABLE_OUTPUTS = (("cable input", "MME"), ("cable in 16 ch", "Windows WASAPI"))
WRITE_SECONDS = 0.05  # tamanho de cada bloco escrito no modo "abre a cada fala"


def _resample(audio: np.ndarray, rate: int, new_rate: int) -> np.ndarray:
    data = np.asarray(audio, dtype=np.float32)
    if rate == new_rate or not len(data):
        return data
    n = max(1, int(round(len(data) * new_rate / rate)))
    return np.interp(np.linspace(0, len(data) - 1, n), np.arange(len(data)), data).astype(np.float32)


def list_output_devices() -> list[tuple[int, str, str]]:
    import sounddevice as sd

    hostapis = sd.query_hostapis()
    return [
        (index, device["name"], hostapis[device["hostapi"]]["name"])
        for index, device in enumerate(sd.query_devices())
        if device["max_output_channels"] > 0
    ]


def find_cable_device() -> int | None:
    outputs = list_output_devices()
    for name, preferred_api in CABLE_OUTPUTS:
        devices = [device for device in outputs if name in device[1].lower()]
        if devices:
            return min(devices, key=lambda device: (device[2] != preferred_api, device[0]))[0]
    return None


def default_output_device() -> int | None:
    import sounddevice as sd

    try:
        device = sd.default.device[1]
        if device is None or device < 0:
            device = sd.query_devices(kind="output")["index"]
        return int(device)
    except Exception:
        return None


class Playback:

    def __init__(self, queued_at: float):
        self.queued_at = queued_at
        self.started: float | None = None


def _finish_stream(stream, *, abort: bool) -> None:
    try:
        if abort:
            stream.abort()
        else:
            stream.stop()
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


class _OpenStream:

    def __init__(self, sd, device: int):
        self.rate = int(sd.query_devices(device)["default_samplerate"])
        self.items: collections.deque = collections.deque()  # itens [áudio, posição já tocada, Playback]
        self.lock = threading.Lock()
        self._drain_until = 0.0  # instante em que o último áudio enviado termina de sair no alto-falante
        self._closed = False
        self.stream = sd.OutputStream(
            device=device, samplerate=self.rate, channels=1, dtype="float32",
            latency="low", callback=self._callback,
        )
        try:
            self.stream.start()
        except Exception:
            _finish_stream(self.stream, abort=True)
            raise

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0]
        out.fill(0)
        if not self.lock.acquire(blocking=False):
            return
        try:
            filled = 0
            now = time.monotonic()
            while filled < frames and self.items:
                audio, pos, playback = self.items[0]
                n = min(frames - filled, len(audio) - pos)
                if n and playback.started is None:
                    playback.started = now
                out[filled:filled + n] = audio[pos:pos + n]
                filled += n
                self.items[0][1] = pos + n
                if pos + n >= len(audio):
                    self.items.popleft()
            if filled:
                # O áudio entregue agora só sai no alto-falante depois da latência do driver.
                latency = max(0.0, time_info.outputBufferDacTime - time_info.currentTime)
                self._drain_until = now + latency + filled / self.rate
        finally:
            self.lock.release()

    def add(self, audio: np.ndarray, rate: int, playback: Playback) -> None:
        data = _resample(audio, rate, self.rate)
        if len(data):
            with self.lock:
                if not self._closed:
                    self.items.append([data, 0, playback])

    def clear(self) -> None:
        with self.lock:
            self.items.clear()

    def busy(self) -> bool:
        with self.lock:
            return bool(self.items) or time.monotonic() < self._drain_until

    def close(self) -> None:
        with self.lock:
            if self._closed:
                return
            self._closed = True
            self.items.clear()
            self._drain_until = 0.0
        _finish_stream(self.stream, abort=True)


@dataclass(eq=False)
class _SimpleJob:

    audio: np.ndarray
    rate: int
    playback: Playback
    cancelled: threading.Event = field(default_factory=threading.Event)


class AudioOut:

    def __init__(self, use_cable: bool = True, use_monitor: bool = True, keep_open: bool = True):
        import sounddevice as sd

        self._sd = sd
        self.cable = find_cable_device() if use_cable else None
        monitor = default_output_device() if use_monitor else None
        self.monitor = monitor if monitor != self.cable else None
        self.targets = [device for device in (self.cable, self.monitor) if device is not None]
        self.muted = False
        self._lock = threading.Lock()
        self._closed = False
        self._open: dict[int, _OpenStream] = {}
        if keep_open:
            for device in self.targets:
                try:
                    self._open[device] = _OpenStream(sd, device)
                except Exception as exc:
                    print(f"[áudio] não foi possível manter {self.device_name(device)} aberto ({exc}); "
                          "ele será aberto a cada fala")

        self._simple_targets = [device for device in self.targets if device not in self._open]
        self._queue: queue.Queue = queue.Queue()
        self._jobs: set[_SimpleJob] = set()  # falas na fila ou tocando (usado por busy())
        self._thread: threading.Thread | None = None
        if self._simple_targets:
            self._thread = threading.Thread(target=self._worker, name="librasfone-audio", daemon=True)
            self._thread.start()

    def device_name(self, index: int | None) -> str:
        if index is None:
            return "-"
        try:
            return self._sd.query_devices(index)["name"]
        except Exception:
            return f"dispositivo {index}"

    def describe(self) -> str:
        cable = f"microfone virtual OK ({self.device_name(self.cable)})" if self.cable is not None else "VB-Cable NÃO encontrado"
        monitor = f"monitor: {self.device_name(self.monitor)}" if self.monitor is not None else "sem monitor"
        mode = "saída sempre aberta" if self._open else "abre a cada fala"
        return f"{cable} | {monitor} | {mode}"

    def play(self, audio: np.ndarray, rate: int) -> Playback | None:
        with self._lock:
            if self._closed or self.muted or not self.targets:
                return None
            data = np.array(audio, dtype=np.float32)  # cópia: quem chamou pode alterar o array original
            if not len(data):
                return None
            playback = Playback(time.monotonic())
            for stream in self._open.values():
                stream.add(data, rate, playback)
            if self._simple_targets:
                job = _SimpleJob(data, rate, playback)
                self._jobs.add(job)
                self._queue.put(job)
            return playback

    def busy(self) -> bool:
        with self._lock:
            return bool(self._jobs) or any(stream.busy() for stream in self._open.values())

    def _cancel_pending(self) -> None:
        for stream in self._open.values():
            stream.clear()
        for job in self._jobs:
            job.cancelled.set()
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                break
            self._jobs.discard(job)
            self._queue.task_done()

    def stop(self) -> None:
        with self._lock:
            if not self._closed:
                self._cancel_pending()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancel_pending()
            if self._thread is not None:
                self._queue.put(None)  # sinal para a thread de trabalho terminar
        for stream in self._open.values():
            stream.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                print("[áudio] o driver ainda não respondeu ao encerramento da saída")

    def _play_on(self, device: int, job: _SimpleJob) -> None:
        error = None
        for use_default in (False, True):
            if job.cancelled.is_set():
                return
            stream = None
            writing = False
            complete = False
            try:
                rate = int(self._sd.query_devices(device)["default_samplerate"]) if use_default else job.rate
                data = _resample(job.audio, job.rate, rate)
                stream = self._sd.OutputStream(device=device, samplerate=rate, channels=1, dtype="float32")
                if job.cancelled.is_set():
                    return
                stream.start()
                chunk = max(1, int(rate * WRITE_SECONDS))
                for pos in range(0, len(data), chunk):
                    if job.cancelled.is_set():
                        return
                    writing = True
                    if job.playback.started is None:
                        job.playback.started = time.monotonic()
                    stream.write(data[pos:pos + chunk].reshape(-1, 1))
                complete = True
                return
            except Exception as exc:
                error = exc
                if writing:
                    break
            finally:
                if stream is not None:
                    _finish_stream(stream, abort=not complete or job.cancelled.is_set())
        if not job.cancelled.is_set():
            print(f"[áudio] falha ao tocar em {self.device_name(device)}: {error}")

    def _worker(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                if not job.cancelled.is_set():
                    # Uma thread por dispositivo, para o VB-Cable e o monitor tocarem juntos.
                    threads = [
                        threading.Thread(target=self._play_on, args=(device, job), daemon=True)
                        for device in self._simple_targets
                    ]
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
            finally:
                with self._lock:
                    self._jobs.discard(job)
                self._queue.task_done()
