"""
Обобщённая линейная регрессия: базовый набор из 5 функций.
Константа, линейная, квадратичная, синусоидальная и экспоненциальная.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "TATN.xlsx"
ASSETS_DIR = BASE_DIR / "assets"


def f(x):
    return 1


def g(x):
    return x


def h(x):
    return x ** 2


def p(x, T):
    return np.sin((2 * np.pi / T) * x)


def r(x, k):
    return np.exp(k * x)


def read_xlsx_to_array(file_path):
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet = workbook.active
    return [
        float(row[1])
        for row in sheet.iter_rows(values_only=True)
        if row[1] is not None and isinstance(row[1], (int, float))
    ]


def calc_coeffs(data, T, k):
    n = len(data)
    t_values = np.arange(1, n + 1)
    funcs = [f, g, h, lambda x: p(x, T), lambda x: r(x, k)]
    func_values = np.array([[func(t) for t in t_values] for func in funcs])
    x_vals = np.dot(func_values, data) / n
    A = np.dot(func_values, func_values.T) / n
    return np.linalg.solve(A, x_vals)


def calculate_rmse(data, coeffs, T, k):
    t_values = np.arange(1, len(data) + 1)
    funcs = [f, g, h, lambda x: p(x, T), lambda x: r(x, k)]
    components = [
        coeff * np.array([func(t) for t in t_values])
        for coeff, func in zip(coeffs, funcs)
    ]
    func_values = sum(components)
    mse = np.mean((data - func_values) ** 2)
    return np.sqrt(mse)


def plot_graph(data_array, coeffs, T, k, title, save_path=None):
    t_values = np.arange(1, len(data_array) + 1)
    funcs = [f, g, h, lambda x: p(x, T), lambda x: r(x, k)]
    components = [
        coeff * np.array([func(t) for t in t_values])
        for coeff, func in zip(coeffs, funcs)
    ]
    func_values = sum(components)

    plt.figure(figsize=(12, 6))
    plt.plot(t_values, data_array, label="Исходные данные", linestyle="-")
    plt.plot(t_values, func_values, label="Линия регрессии", linestyle="--")
    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("Цена")
    plt.grid()
    plt.legend()
    if save_path is not None:
        ASSETS_DIR.mkdir(exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    data_array = read_xlsx_to_array(DATA_PATH)
    T = 40
    k = 0.01

    coeffs = calc_coeffs(data_array, T, k)
    print("Коэффициенты:", *coeffs)

    rmse = calculate_rmse(data_array, coeffs, T, k)
    print(f"Среднеквадратичная ошибка (RMSE): {rmse:.2f}")

    plot_graph(
        data_array,
        coeffs,
        T,
        k,
        "Обобщённая регрессия для цен акций (базовая модель)",
        save_path=ASSETS_DIR / "regression_basic.png",
    )


if __name__ == "__main__":
    main()
