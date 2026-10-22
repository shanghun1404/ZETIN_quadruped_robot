"""Pull authorized HPC job results and record their actual state; no simulation.

This local observer submits no jobs and changes no HPC source, controller,
baseline, git remote or Drive contents. It runs until the fixed pipeline has
finished, or its bounded observation window expires.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "reports/v16"
HPC = "/home/shanghun1404/v16_12dof_model"
JOBS = {563: "physics_jobs", 568: "physics_jobs", 570: "comparison_jobs",
        571: "physics_jobs", 582: "evaluation_jobs", 573: "evaluation_jobs", 583: "video_jobs",
        575: "physics_jobs", 576: "physics_jobs", 577: "physics_jobs",
        578: "evaluation_jobs", 579: "evaluation_jobs", 584: "video_jobs"}
EVALUATION_SECONDS = {582: 12., 573: 32., 578: 12., 579: 32.}
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-1000:] or result.stdout.strip()[-1000:])
    return result.stdout


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None


def required_json(path):
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object is missing or invalid: {path}")
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pull(remote, local):
    local.mkdir(parents=True, exist_ok=True)
    command(["rsync", "-az", "-e", "ssh -o ClearAllForwardings=yes -o ConnectTimeout=10",
             "ws1:" + remote + "/", str(local) + "/"])


def observe():
    output = command(["ssh", "-o", "ClearAllForwardings=yes", "-o", "ConnectTimeout=10", "ws1",
                      "sacct -nP -j " + ",".join(map(str, JOBS)) + " --format=JobIDRaw,State,Elapsed,ExitCode"])
    states = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 4 and fields[0].isdigit() and int(fields[0]) in JOBS:
            states[int(fields[0])] = dict(state=fields[1].split()[0].rstrip("+"),
                elapsed=fields[2], exit_code=fields[3])
    snapshot = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                    updated_kst=datetime.now(ZoneInfo("Asia/Seoul")).isoformat(), jobs={},
                    walking_criteria_passed=None, wbc_walking_criteria_passed=None, errors=[])
    for job, folder in JOBS.items():
        record = states.get(job, dict(state="UNKNOWN", elapsed="", exit_code=""))
        snapshot["jobs"][str(job)] = record
        if record["state"] in ("PENDING", "UNKNOWN"):
            continue
        local = OUT / folder / str(job)
        try:
            pull(f"{HPC}/{folder}/{job}", local)
            if record["state"] == "COMPLETED":
                sources = OUT / "source_snapshots" / str(job)
                pull(f"{HPC}/job_sources/{job}", sources)
                checkpoint = read_json(local / "checkpoint.json")
                if checkpoint:
                    expected = checkpoint["controller_code_sha256"]
                    source_ok = all((sources/name).is_file() and sha(sources/name) == digest
                                    for name, digest in expected.items())
                    manifest_ok = sha(local / "manifest.json") == checkpoint["manifest_sha256"]
                    record["source_and_manifest_verified"] = source_ok and manifest_ok
                    if not record["source_and_manifest_verified"]:
                        raise ValueError("controller or manifest hash mismatch")
                    record["generations_completed"] = checkpoint["generation"] + 1
                    record["best_parameters_qualified"] = checkpoint.get("best_parameters_qualified")
            checkpoint = read_json(local / "checkpoint.json")
            if checkpoint and checkpoint["history"]:
                record["latest_training"] = checkpoint["history"][-1]
            summary = read_json(local / "holdout_summary.json")
            if summary:
                record["evaluation_rows"] = summary["rows"]
                expected_seconds = EVALUATION_SECONDS[job]
                if record["state"] == "COMPLETED" and record["exit_code"] == "0:0":
                    conditions = required_json(local / "evaluation_conditions.json")
                    held_checkpoint = required_json(local / "training_checkpoint.json")
                    source_ok = all((sources/name).is_file() and sha(sources/name) == digest
                        for name, digest in held_checkpoint["controller_code_sha256"].items())
                    manifest_ok = sha(local / "training_manifest.json") == held_checkpoint["manifest_sha256"]
                    reports = [required_json(path) for path in local.glob("model_*/eval_seed*.json")]
                    duration_ok = (conditions["evaluation_seconds"] == expected_seconds
                        and summary["evaluation_seconds"] == expected_seconds
                        and len(reports) == 3 and {r["seed"] for r in reports} == {1001,1002,1003}
                        and all(r["seconds"] == expected_seconds for r in reports))
                    record["source_and_conditions_verified"] = (source_ok and manifest_ok and duration_ok
                        and conditions["holdout_seeds_disjoint"] and summary["source_and_conditions_verified"])
                    if not record["source_and_conditions_verified"]:
                        raise ValueError("holdout source/conditions/duration mismatch")
                    record["walking_criteria_passed"] = summary["model_holdout_passed"]
            if folder == "video_jobs":
                record["videos"] = []
                if record["state"] == "COMPLETED" and record["exit_code"] == "0:0":
                    video_checkpoint = required_json(local / "training_checkpoint.json")
                    video_manifest = required_json(local / "manifest.json")
                    source_ok = (video_manifest["code_sha256"] == video_checkpoint["controller_code_sha256"]
                        and video_manifest["initial_parameters_sha256"] == sha(local / "training_checkpoint.json")
                        and all((sources/name).is_file() and sha(sources/name) == digest
                            for name,digest in video_checkpoint["controller_code_sha256"].items()))
                    if not source_ok:
                        raise ValueError("trained video source/parameter mismatch")
                    record["source_and_parameters_verified"] = True
                    for path in local.rglob("*.mp4"):
                        if path.stat().st_size:
                            metadata = json.loads(command(["ffprobe", "-v", "error", "-show_entries",
                                "format=duration:stream=width,height,nb_frames", "-of", "json", str(path)]))
                            record.setdefault("video_metadata", []).append(metadata)
                            record["videos"].append(str(path.relative_to(OUT)))
                    if not record["videos"]:
                        raise ValueError("completed video job contains no nonempty MP4")
        except (OSError, KeyError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            snapshot["errors"].append(f"{job}: {error}")
    evaluations = [snapshot["jobs"][str(job)].get("walking_criteria_passed") for job in (582, 573)]
    if not snapshot["errors"] and all(value is not None for value in evaluations):
        snapshot["walking_criteria_passed"] = all(evaluations)
    wbc_evaluations = [snapshot["jobs"][str(job)].get("walking_criteria_passed") for job in (578, 579)]
    if not snapshot["errors"] and all(value is not None for value in wbc_evaluations):
        snapshot["wbc_walking_criteria_passed"] = all(wbc_evaluations)
    write(snapshot)
    return snapshot


def write(snapshot):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = OUT / "overnight_status.tmp"
    temp.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    temp.replace(OUT / "overnight_status.json")
    lines = ["# V16 야간 훈련·평가 상태", "", f"마지막 확인(KST): {snapshot['updated_kst']}", "",
             "HPC의 실제 Slurm 상태와 저장 결과를 로컬로 복사한 기록이다. 진행 중인 훈련 점수는 독립 평가 결과가 아니다.", "",
             "| Job | 작업 | 상태 | 실행시간 | 종료코드 |", "|---|---|---|---|---|"]
    for job, record in snapshot["jobs"].items():
        lines.append(f"| {job} | {JOBS[int(job)]} | {record['state']} | {record['elapsed']} | {record['exit_code']} |")
    lines.extend(["", "## 훈련 진행", ""])
    for job, record in snapshot["jobs"].items():
        if "latest_training" in record:
            row = record["latest_training"]
            lines.extend([f"- {job}: generation {row['generation']}, 생존 {row['survived']}, "
                f"선택 후보의 진행 {row['progress_best_m']:.3f}m, 완료 steps {row.get('completed_step_count_best','미기록')}."])
    label = {None: "평가 미완료", True: "통과", False: "미통과"}[snapshot["walking_criteria_passed"]]
    wbc_label = {None: "평가 미완료", True: "통과", False: "미통과"}[snapshot["wbc_walking_criteria_passed"]]
    lines.extend(["", "## 독립 평가", "", f"접촉력 QP의12초·32초 보행 기준: **{label}**.", "",
                  f"가속도·접촉력 WBC의12초·32초 보행 기준: **{wbc_label}**.", "",
                  "기준은 생존95% 이상, 평균 속도 오차0.05m/s 이하, 평균 진행이 명령 거리의75–125%, 유한 통계다.",
                  "1001–1003 seed는 QP 훈련101–312와 WBC 훈련401–528에서 분리한다. 같은 고정 모델을12초와32초에서 평가하며 평가 사이에 재튜닝하지 않는다.",
                  "두 길이와 두 제어기는 같은 평가 seed를 사용하는 비교이므로 서로 독립적인 시행으로 합산하지 않는다.",
                  "QP의 유한 반복 제약 위반과 별도 토크 guard는 각 평가 JSON에 기록한다. 보행 기준 통과가 모든 QP 제약의 만족을 뜻하지 않는다."])
    for job in EVALUATION_SECONDS:
        record = snapshot["jobs"].get(str(job), {})
        if "walking_criteria_passed" in record:
            lines.extend(["", f"- {job}: [{record['walking_criteria_passed']}]({JOBS[job]}/{job}/README.md)"])
    videos = sum((snapshot["jobs"].get(str(job), {}).get("videos", []) for job in (583,584)), [])
    lines.extend(["", "## 영상", ""] + [f"- [{Path(path).name}]({path})" for path in videos])
    if not videos:
        lines.append("583/584이 끝나면 관찰 카메라 영상과 실제 길이·프레임 수가 복사된다(요청24초, 실패 시 조기에 끝날 수 있음).")
    lines.extend(["", "V15의 추가 토크 최적화 후보는 채택하지 않았고339 원본을 유지한다. V16은 독립적인 물리 제어 실험이다."])
    if snapshot["errors"]:
        lines.extend(["", "## 동기화 오류", ""] + snapshot["errors"])
    temp_md = OUT / "overnight_status.md.tmp"
    temp_md.write_text("\n".join(lines) + "\n")
    temp_md.replace(OUT / "overnight_status.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=120.)
    parser.add_argument("--hours", type=float, default=6.)
    args = parser.parse_args()
    deadline = time.monotonic() + args.hours * 3600
    while True:
        try:
            result = observe()
            print(result["updated_utc"], {job:r["state"] for job,r in result["jobs"].items()}, flush=True)
            done = all(r["state"] in TERMINAL for r in result["jobs"].values()) and not result["errors"]
        except (OSError, KeyError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(datetime.now(timezone.utc).isoformat(), str(error), flush=True)
            done = False
        if args.once or done or time.monotonic() >= deadline:
            break
        time.sleep(args.interval)
