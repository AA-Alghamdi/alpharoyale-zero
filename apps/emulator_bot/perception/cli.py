"""CLI: perceive a frame and print the unified JSON.

    python -m perception.cli <image> [backend]
"""
import sys

from perception.base import available, get_perceptor


def main() -> None:
    if len(sys.argv) < 2:
        print(f"usage: python -m perception.cli <image> [backend]  "
              f"(backends: {available() or ['buildabot']})")
        sys.exit(1)
    image = sys.argv[1]
    backend = sys.argv[2] if len(sys.argv) > 2 else "buildabot"
    perceptor = get_perceptor(backend)
    print(perceptor.detect(image).to_json(indent=2))


if __name__ == "__main__":
    main()
