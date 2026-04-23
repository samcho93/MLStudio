"""100개 예제 자동 실행 테스트 — 에러 수집 및 보고 (CUDA 복구 포함)."""
import asyncio
import json
import os
import sys
import time
import traceback

# 타임아웃 (초) — 예제당 최대 실행 시간
TIMEOUT_PER_EXAMPLE = 120


def _cuda_cleanup():
    """CUDA 상태 초기화 — 예제 간 캐스케이딩 에러 방지."""
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            # CUDA 에러 상태 리셋 시도
            try:
                _ = torch.zeros(1, device="cuda")
                del _
            except Exception:
                pass
    except Exception:
        pass

    # matplotlib 메모리 정리
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass

    try:
        import gc
        gc.collect()
    except Exception:
        pass


async def run_example(example_path: str) -> dict:
    """단일 예제 실행. 결과 dict 반환."""
    with open(example_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    example_id = meta.get("id", "?")
    title = meta.get("title", os.path.basename(example_path))

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    if not nodes:
        return {
            "id": example_id,
            "title": title,
            "status": "skip",
            "message": "No nodes",
            "time": 0,
        }

    from pipeline_executor import PipelineExecutor

    messages = []

    async def ws_cb(msg):
        messages.append(msg)

    executor = PipelineExecutor(nodes, edges, ws_cb)

    start = time.time()
    try:
        await asyncio.wait_for(executor.execute(), timeout=TIMEOUT_PER_EXAMPLE)
        elapsed = round(time.time() - start, 2)

        # 학습 완료 확인
        completed = any(m.get("type") == "TRAINING_COMPLETE" for m in messages)
        pipeline_done = any(m.get("type") == "PIPELINE_COMPLETE" for m in messages)

        return {
            "id": example_id,
            "title": title,
            "status": "ok",
            "message": f"{'trained' if completed else 'pipeline'} ok",
            "time": elapsed,
        }
    except asyncio.TimeoutError:
        elapsed = round(time.time() - start, 2)
        return {
            "id": example_id,
            "title": title,
            "status": "timeout",
            "message": f"Timeout after {TIMEOUT_PER_EXAMPLE}s",
            "time": elapsed,
        }
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        tb = traceback.format_exc()
        # 핵심 에러 메시지만 추출
        short_tb = "\n".join(tb.strip().split("\n")[-3:])
        return {
            "id": example_id,
            "title": title,
            "status": "error",
            "message": str(e)[:200],
            "traceback": short_tb,
            "time": elapsed,
        }
    finally:
        # 예제 간 CUDA 상태 정리
        _cuda_cleanup()


async def main():
    examples_dir = "examples"
    example_files = []

    for root, dirs, files in os.walk(examples_dir):
        for fname in sorted(files):
            if fname.endswith(".json") and fname != "_index.json":
                example_files.append(os.path.join(root, fname))

    example_files.sort()
    total = len(example_files)
    print(f"Found {total} examples to test\n")
    print("=" * 80)

    results = []
    ok_count = 0
    err_count = 0
    skip_count = 0

    for i, path in enumerate(example_files):
        fname = os.path.basename(path)
        short = fname[:40].ljust(40)

        result = await run_example(path)
        results.append(result)

        status = result["status"]
        eid = str(result["id"]).rjust(3)
        elapsed = f"{result['time']:.1f}s".rjust(7)

        if status == "ok":
            ok_count += 1
            icon = "OK"
            print(f"  [{eid}] {icon}  {elapsed}  {short}")
        elif status == "skip":
            skip_count += 1
            icon = "SKIP"
            print(f"  [{eid}] {icon} {elapsed}  {short}  -- {result['message']}")
        elif status == "timeout":
            err_count += 1
            icon = "TIME"
            print(f"  [{eid}] {icon} {elapsed}  {short}  -- {result['message']}")
        else:
            err_count += 1
            icon = "FAIL"
            msg = result["message"][:60]
            print(f"  [{eid}] {icon} {elapsed}  {short}  -- {msg}")

    # 요약
    print("\n" + "=" * 80)
    print(f"TOTAL: {total}  |  OK: {ok_count}  |  FAIL: {err_count}  |  SKIP: {skip_count}")
    print("=" * 80)

    # 실패 목록 상세
    failures = [r for r in results if r["status"] in ("error", "timeout")]
    if failures:
        print(f"\n{'='*80}")
        print(f"FAILURES ({len(failures)}):")
        print(f"{'='*80}")
        for r in failures:
            print(f"\n--- Example {r['id']}: {r['title']} ---")
            print(f"  Status: {r['status']}")
            print(f"  Error: {r['message']}")
            if "traceback" in r:
                print(f"  Traceback:\n    {r['traceback']}")

    # JSON 결과 저장
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to test_results.json")

    return err_count


if __name__ == "__main__":
    err = asyncio.run(main())
    sys.exit(1 if err > 0 else 0)
