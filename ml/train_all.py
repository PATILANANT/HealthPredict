"""Train every module in one go:  python -m ml.train_all [--force] [--fast]"""
import sys

from ml.diabetes import train as diabetes
from ml.mental_health import train as mental
from ml.stroke import train as stroke


def main():
    force = ["--force"] if "--force" in sys.argv else []
    diabetes.main(force)
    stroke.main(force)
    mental.main(force + (["--fast"] if "--fast" in sys.argv else []))


if __name__ == "__main__":
    main()
