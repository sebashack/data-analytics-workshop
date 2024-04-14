import sys

from menu import (
    choose_operation,
    create_parser,
)

def main():
    parser = create_parser()
    args = parser.parse_args()
    choose_operation(args)

if __name__ == "__main__":
    sys.exit(main())
