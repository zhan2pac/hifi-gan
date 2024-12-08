from pathlib import Path

import gdown

URLS = {"https://clck.ru/3F7ReC": "saved/hifigan_v1_lrcos/model_best.pth"}


def main():
    path_gzip = Path("saved/").absolute().resolve()
    path_gzip.mkdir(exist_ok=True, parents=True)

    for url, path in URLS.items():
        gdown.download(url, path)
        print("Model downloaded to", str(Path(path).absolute().resolve()))


if __name__ == "__main__":
    main()
