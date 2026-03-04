from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from sklearn import metrics, svm

if TYPE_CHECKING:
    from numpy.typing import NDArray


def log(message: str) -> None:
    print(f"[part2] {message}")  # noqa: T201


def should_log_progress(index: int, total: int) -> bool:
    if total <= 0:
        return False

    current = index + 1
    if current == total:
        return True

    percent = (current * 100) // total
    previous_percent = (index * 100) // total
    return percent // 10 > previous_percent // 10


def load_mnist(path: Path, kind: str = "train") -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    log(f"load_mnist(kind='{kind}')")

    labels_path = path / f"{kind}-labels-idx1-ubyte.gz"
    images_path = path / f"{kind}-images-idx3-ubyte.gz"

    with gzip.open(labels_path, "rb") as labels_file:
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as images_file:
        images = np.frombuffer(images_file.read(), dtype=np.uint8, offset=16).reshape(
            len(labels),
            784,
        )

    log(f"loaded {len(labels)} samples for kind='{kind}'")
    return images, labels


def extract_hog(image: NDArray[np.uint8]) -> NDArray[np.float64]:
    hog_feature = feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys",
    )
    return cast("NDArray[np.float64]", hog_feature)


def save_confusion_matrix_image(
    confusion: NDArray[np.int64],
    accuracy: float,
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 9))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()

    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            plt.text(
                col_index,
                row_index,
                f"{confusion[row_index, col_index]:d}",
                ha="center",
                va="center",
            )

    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.title(f"Accuracy Score: {accuracy:.4f}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_sample_predictions(
    x_test: NDArray[np.uint8],
    y_test: NDArray[np.uint8],
    model: svm.SVC,
    label_names: list[str],
    output_path: Path,
) -> None:
    rng = np.random.default_rng(seed=42)
    sample_indices = rng.choice(np.arange(0, len(y_test)), size=16, replace=False)

    images: list[NDArray[np.uint8]] = []
    for index in sample_indices:
        test_image = x_test[index]
        hog_feature = extract_hog(test_image)
        predicted = int(model.predict(hog_feature.reshape(1, -1))[0])
        predicted_label = label_names[predicted]
        true_label = label_names[int(y_test[index])]

        canvas = cv2.merge([test_image] * 3)
        canvas = cv2.resize(canvas, (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(
            canvas,
            f"P:{predicted_label}",
            (2, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )
        cv2.putText(
            canvas,
            f"T:{true_label}",
            (2, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )
        images.append(canvas)

    grid_rows = [np.hstack(images[row : row + 4]) for row in range(0, 16, 4)]
    grid = np.vstack(grid_rows)

    cv2.imwrite(str(output_path), grid)


def main() -> None:  # noqa: PLR0915
    log("main() start")

    # Step 1) 데이터셋 경로 준비
    data_dir = Path(__file__).resolve().parent / "fashion"
    if not data_dir.exists():
        msg = f"Fashion-MNIST 경로를 찾을 수 없습니다: {data_dir}"
        raise FileNotFoundError(msg)

    # Step 2) Fashion-MNIST 데이터 로드 (train / test)
    log("Step 2: load train/test dataset")
    x_train, y_train = load_mnist(data_dir, kind="train")
    x_test, y_test = load_mnist(data_dir, kind="t10k")

    label_names = [
        "top",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]

    # Step 3) 784 벡터를 28x28 이미지로 복원
    log("Step 3: reshape vectors to 28x28")
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    # Step 4) 학습 데이터 HOG 특징 추출
    log("Step 4: extract HOG features for training set")
    data_train: list[NDArray[np.float64]] = []
    labels_train: list[int] = []

    train_total = len(x_train)
    for idx, image in enumerate(x_train):
        hog_feature = extract_hog(image)
        data_train.append(hog_feature)
        labels_train.append(int(y_train[idx]))

        if should_log_progress(idx, train_total):
            percent = ((idx + 1) * 100) // train_total
            log(f"Step 4 progress: {min(percent, 100)}% ({idx + 1}/{train_total})")

    x_train_hog = np.asarray(data_train)
    y_train_hog = np.asarray(labels_train)

    # Step 5) SVM(RBF, C=100, random_state=42) 학습
    log("Step 5: train SVM classifier")
    model = svm.SVC(kernel="rbf", C=100, random_state=42)
    model.fit(x_train_hog, y_train_hog)

    train_accuracy = model.score(x_train_hog, y_train_hog)
    log(f"Train set accuracy: {train_accuracy:.4f}")

    # Step 6) 테스트 데이터 HOG 특징 추출 및 예측
    log("Step 6: extract HOG features for test set and predict")
    data_test: list[NDArray[np.float64]] = []

    test_total = len(x_test)
    for idx, image in enumerate(x_test):
        hog_feature = extract_hog(image)
        data_test.append(hog_feature)

        if should_log_progress(idx, test_total):
            percent = ((idx + 1) * 100) // test_total
            log(f"Step 6 progress: {min(percent, 100)}% ({idx + 1}/{test_total})")

    x_test_hog = np.asarray(data_test)
    predictions = model.predict(x_test_hog)

    # Step 7) 정확도/혼동행렬 계산 및 저장
    test_accuracy = metrics.accuracy_score(y_test, predictions)
    confusion = metrics.confusion_matrix(y_test, predictions)

    log(f"Test set accuracy: {test_accuracy:.4f}")
    log("Confusion matrix:")
    log(confusion)

    confusion_path = Path(__file__).resolve().parent / "part2_confusion_matrix.png"
    save_confusion_matrix_image(confusion, test_accuracy, confusion_path)
    log(f"Saved confusion matrix image: {confusion_path}")

    # Step 8) 샘플 예측 시각화 저장
    sample_path = Path(__file__).resolve().parent / "part2_sample_predictions.png"
    save_sample_predictions(x_test, y_test, model, label_names, sample_path)
    log(f"Saved sample prediction image: {sample_path}")

    log("main() done")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
