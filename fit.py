import pandas as pd
from prepare_data import HCDALoader

def main():
    loader = HCDALoader()
    applications_train = loader.read_applications_train()
    bureau_summary = loader.read_bureau()
    previous_summary = loader.read_previous_application()

if __name__ == "__main__":
    main()
