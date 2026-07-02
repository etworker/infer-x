"""Multi-source model downloader (HuggingFace / ModelScope / HF-Mirror / URL)."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .models import DownloadProgress, DownloadRequest, DownloadSource, DownloadStatus


class ModelDownloader:
    def __init__(self, model_dir: str, hf_mirror_url: Optional[str] = None, max_concurrent: int = 2):
        self._model_dir = Path(model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)
        self._hf_mirror_url = hf_mirror_url
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: Dict[str, DownloadProgress] = {}

    @property
    def tasks(self) -> Dict[str, DownloadProgress]:
        return dict(self._tasks)

    def get_task(self, task_id: str) -> Optional[DownloadProgress]:
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a download task."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status in (DownloadStatus.completed, DownloadStatus.failed):
            return False
        task.status = DownloadStatus.failed
        task.error = "Cancelled by user"
        return True

    async def start_download(self, req: DownloadRequest) -> DownloadProgress:
        task_id = uuid.uuid4().hex[:12]
        progress = DownloadProgress(
            task_id=task_id,
            source=req.source.value,
            repo=req.repo,
            filename=req.filename,
            status=DownloadStatus.pending,
        )
        self._tasks[task_id] = progress
        asyncio.create_task(self._run_download(task_id, req))
        return progress

    async def _run_download(self, task_id: str, req: DownloadRequest) -> None:
        progress = self._tasks[task_id]
        try:
            async with self._semaphore:
                progress.status = DownloadStatus.downloading
                if req.source == DownloadSource.huggingface:
                    await self._download_hf(progress, req)
                elif req.source == DownloadSource.hf_mirror:
                    await self._download_hf(progress, req, mirror=True)
                elif req.source == DownloadSource.modelscope:
                    await self._download_modelscope(progress, req)
                elif req.source == DownloadSource.url:
                    await self._download_url(progress, req)
                progress.status = DownloadStatus.completed
                progress.progress_pct = 100.0
        except Exception as e:
            progress.status = DownloadStatus.failed
            progress.error = str(e)

    # ---- HuggingFace -------------------------------------------------------

    async def _download_hf(
        self, progress: DownloadProgress, req: DownloadRequest, mirror: bool = False
    ) -> None:
        from huggingface_hub import hf_hub_download

        repo = req.repo or ""
        filename = req.filename
        if "/" not in repo:
            raise ValueError("repo must be in 'user/repo' format")

        env = os.environ.copy()
        if mirror and self._hf_mirror_url:
            env["HF_ENDPOINT"] = self._hf_mirror_url

        def _download() -> str:
            kwargs: Dict[str, Any] = {"repo_id": repo}
            if filename:
                kwargs["filename"] = filename
            if req.quantization:
                kwargs["revision"] = req.quantization
            return hf_hub_download(**kwargs, local_dir=str(self._model_dir), force_download=True)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _download)
        progress.save_path = str(result)
        progress.progress_pct = 100.0

    # ---- ModelScope --------------------------------------------------------

    async def _download_modelscope(self, progress: DownloadProgress, req: DownloadRequest) -> None:
        repo = req.repo or ""
        filename = req.filename
        if "/" not in repo:
            raise ValueError("repo must be in 'user/repo' format")

        def _download() -> str:
            try:
                from modelscope.hub.snapshot_download import snapshot_download as ms_download
            except ImportError:
                raise ImportError("modelscope not installed. Run: pip install modelscope")
            result = ms_download(
                repo_id=repo,
                local_dir=str(self._model_dir / repo.split("/")[-1]),
            )
            if filename:
                return str(Path(result) / filename)
            return str(result)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _download)
        progress.save_path = result
        progress.progress_pct = 100.0

    # ---- URL download ------------------------------------------------------

    async def _download_url(self, progress: DownloadProgress, req: DownloadRequest) -> None:
        import httpx

        url = req.url
        if not url:
            raise ValueError("url is required for url source")

        save_name = req.save_name or url.split("/")[-1].split("?")[0]
        save_path = self._model_dir / save_name

        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                progress.total_bytes = total
                downloaded = 0
                start_time = time.time()
                with open(save_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress.downloaded_bytes = downloaded
                        if total > 0:
                            progress.progress_pct = round(downloaded / total * 100, 1)
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            progress.speed_bytes_per_sec = downloaded / elapsed

        progress.save_path = str(save_path)

    # ---- Auto-download safetensor for gguf models --------------------------

    async def auto_download_safetensors(self, gguf_models: list, source: str = "hf") -> list:
        """Auto-download safetensor versions for gguf models.
        
        Args:
            gguf_models: List of gguf model filenames
            source: "hf" for HuggingFace, "ms" for ModelScope
        """
        results = []
        
        # Map gguf model names to model repos
        hf_model_map = {
            "qwen2.5": "Qwen/Qwen2.5-{size}-Instruct",
            "qwen3": "Qwen/Qwen3-{size}",
            "qwen3.5": "Qwen/Qwen3.5-{size}",
            "gemma-4": "google/gemma-4-{size}-it",
            "llama": "meta-llama/Llama-{size}",
            "mistral": "mistralai/Mistral-{size}",
        }
        
        ms_model_map = {
            "qwen2.5": "Qwen/Qwen2.5-{size}-Instruct",
            "qwen3": "Qwen/Qwen3-{size}",
            "qwen3.5": "Qwen/Qwen3.5-{size}",
            "gemma-4": "AI-ModelScope/gemma-4-{size}-it",
            "llama": "meta-llama/Llama-{size}",
            "mistral": "mistralai/Mistral-{size}",
        }
        
        model_map = ms_model_map if source == "ms" else hf_model_map
        
        for gguf_name in gguf_models:
            if not gguf_name.endswith(".gguf"):
                continue
                
            name_lower = gguf_name.lower()
            repo = None
            
            for family, repo_template in model_map.items():
                if family in name_lower:
                    import re
                    size_match = re.search(r'(\d+\.?\d*[bB])', gguf_name)
                    if size_match:
                        size = size_match.group(1).replace('B', 'b').replace('b', 'B')
                        repo = repo_template.replace("{size}", size)
                    break
            
            if not repo:
                clean_name = re.sub(r'-Q[0-9]+_[A-Z0-9]+\.gguf$', '', gguf_name, flags=re.IGNORECASE)
                clean_name = re.sub(r'-q[0-9]+_[a-z0-9]+\.gguf$', '', clean_name, flags=re.IGNORECASE)
                if "qwen" in clean_name.lower():
                    repo = f"Qwen/{clean_name}"
                elif "gemma" in clean_name.lower():
                    repo = f"AI-ModelScope/{clean_name}" if source == "ms" else f"google/{clean_name}"
            
            if not repo:
                results.append({"gguf": gguf_name, "status": "skipped", "reason": "cannot determine repo"})
                continue
            
            repo_name = repo.split("/")[-1]
            target_dir = self._model_dir / repo_name
            if target_dir.exists() and any(target_dir.glob("*.safetensors")):
                results.append({"gguf": gguf_name, "repo": repo, "status": "exists", "path": str(target_dir)})
                continue
            
            try:
                if source == "ms":
                    result = await self._download_modelscope_auto(repo, repo_name)
                else:
                    result = await self._download_hf_auto(repo, repo_name)
                results.append({"gguf": gguf_name, "repo": repo, "status": "downloaded", "path": result})
            except Exception as e:
                results.append({"gguf": gguf_name, "repo": repo, "status": "error", "error": str(e)})
        
        return results

    async def _download_hf_auto(self, repo: str, repo_name: str) -> str:
        from huggingface_hub import snapshot_download
        import os
        
        env = os.environ.copy()
        if self._hf_mirror_url:
            env["HF_ENDPOINT"] = self._hf_mirror_url
        
        def _download():
            return snapshot_download(
                repo_id=repo,
                local_dir=str(self._model_dir / repo_name),
            )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download)

    async def _download_modelscope_auto(self, repo: str, repo_name: str) -> str:
        def _download():
            try:
                from modelscope.hub.snapshot_download import snapshot_download as ms_download
            except ImportError:
                raise ImportError("modelscope not installed")
            return ms_download(
                repo_id=repo,
                local_dir=str(self._model_dir / repo_name),
            )
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _download)
