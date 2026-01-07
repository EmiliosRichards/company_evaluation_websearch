def main() -> int:
    # Backwards-compatible shim.
    from scripts.evaluate_list import main as _main

    return _main()


if __name__ == "__main__":
    raise SystemExit(main())


