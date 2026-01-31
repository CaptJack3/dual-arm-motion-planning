import os
import shutil
import traceback
import faulthandler

from Experiment_last_working_28_1 import Experiment


def run_attempts(n=10):
    faulthandler.enable()

    for k in range(1, n + 1):
        print(f"\n================ Attempt {k}/{n} ================\n")

        # Optional: isolate outputs per attempt (recommended)
        attempt_dir = os.path.join("outputs", f"attempt_{k:02d}")
        os.makedirs(attempt_dir, exist_ok=True)

        # If your code always writes to ./outputs/, we can temporarily redirect by copying afterward.
        # Easiest approach: clear outputs/ before run, then copy to attempt folder.
        os.makedirs("outputs", exist_ok=True)
        for fname in os.listdir("outputs"):
            path = os.path.join("outputs", fname)
            # don't delete attempt folders
            if os.path.isdir(path) and fname.startswith("attempt_"):
                continue
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except Exception:
                pass

        try:
            exp = Experiment()
            exp.plan_experiment()

            print(f"\n✅ Success on attempt {k}!")
            # Copy resulting files into attempt_dir (so you keep them)
            for fname in os.listdir("outputs"):
                src = os.path.join("outputs", fname)
                if os.path.isdir(src) and fname.startswith("attempt_"):
                    continue
                dst = os.path.join(attempt_dir, fname)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

            return True  # stop after first success

        except Exception as e:
            print(f"\n❌ Attempt {k} failed with error:\n{e}\n")
            traceback.print_exc()

            # Save whatever got produced (logs/plots) into attempt dir
            for fname in os.listdir("outputs"):
                src = os.path.join("outputs", fname)
                if os.path.isdir(src) and fname.startswith("attempt_"):
                    continue
                dst = os.path.join(attempt_dir, fname)
                if os.path.isfile(src):
                    try:
                        shutil.copy2(src, dst)
                    except Exception:
                        pass

            continue

    print("\n❌ All attempts failed.")
    return False


if __name__ == "__main__":
    run_attempts(n=10)
