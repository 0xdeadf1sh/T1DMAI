#!/usr/bin/env bash
# Export a checkpoint to ExecuTorch XNNPACK and adb push it into T1DMDROID's models dir.
set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: $0 <checkpoint.pt> [model-id]" >&2; exit 2; }

ckpt=$(realpath "$1")
[[ -f $ckpt ]] || { echo "no checkpoint: $1" >&2; exit 1; }
# a push overwrites its id on the phone, so only models/<capacity>/checkpoints/ implies an id
if [[ -n ${2:-} ]]; then
    id=$2
elif [[ $ckpt =~ /models/([^/]+)/checkpoints/[^/]+$ ]]; then
    id=${BASH_REMATCH[1]}
else
    echo "$1 is outside models/<capacity>/checkpoints/; pass a model-id" >&2
    exit 2
fi

root=$(cd "$(dirname "$0")" && pwd)
out=$root/exported/$id
remote=/sdcard/Android/data/com.t1dm.app/files/models

cd "$root"
.venv-export/bin/python -m exporters.executorch_xnnpack --checkpoint "$ckpt" --model-id "$id" --out-dir "$out"

[[ $(adb get-state 2>/dev/null) == device ]] || { echo "no device; push skipped, export in $out" >&2; exit 1; }

adb shell mkdir -p "$remote"
# descriptor last: it is what ModelStore discovers
for f in "$id.xnnpack.pte" "$id.head.bin" "$id.xnnpack.descriptor.json"; do
    adb push "$out/$f" "$remote/$f"
    local_size=$(stat -c %s "$out/$f")
    remote_size=$(adb shell stat -c %s "$remote/$f" | tr -d '\r')
    [[ $local_size == "$remote_size" ]] || { echo "$f: $remote_size/$local_size bytes on device" >&2; exit 1; }
done

echo "pushed $id; loads on next app start"
