import kagglehub


def main():
    path = kagglehub.dataset_download(
        "khanfashee/nih-chest-x-ray-14-224x224-resized"
    )

    print(
        "NIH 224 dataset path:"
    )

    print(
        path
    )


if __name__ == "__main__":
    main()