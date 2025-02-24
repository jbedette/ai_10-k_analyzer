import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augmentation Pipeline")
    parser.add_argument("--gpu",action="store_true", help="User GPU if available")
    parser.add_argument("--cpu", action="store_true", help="Use CPU ThreadPool instead of ProcessPool")
    parser.add_argument("--fast", action="store_true", help="Load lowest weight text transform models")
    parser.add_argument("--med", action="store_true", help="Load middling weight text transform models")
    parser.add_argument("--slow", action="store_true", help="Load heaviest weight text transform models")
    args= parser.parse_args()
    print(args)

