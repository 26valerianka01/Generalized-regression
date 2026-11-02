"""
Обобщённая линейная регрессия: расширенный набор базисных функций.
Полиномы до 4-й степени, несколько синусоид и экспонента.
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


def p(x, omega):
    return np.sin(omega * x)


def r(x, k):
    return np.exp(k * x)


def s(x):
    return x ** 3


def t(x):
    return x ** 4


def read_xlsx_to_array(file_path):
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    sheet = workbook.active
    return [
        float(row[1])
        for row in sheet.iter_rows(values_only=True)
        if row[1] is not None and isinstance(row[1], (int, float))
    ]


def calc_coeffs(data, funcs):
    n = len(data)
    t_values = np.arange(1, n + 1)
    func_values = np.array([[func(t) for t in t_values] for func in funcs])
    A = np.dot(func_values, func_values.T) / n
    x_vals = np.dot(func_values, data) / n
    return np.linalg.solve(A, x_vals)


def calculate_rmse(data, coeffs, funcs):
    t_values = np.arange(1, len(data) + 1)
    components = [
        coeff * np.array([func(t) for t in t_values])
        for coeff, func in zip(coeffs, funcs)
    ]
    func_values = sum(components)
    mse = np.mean((data - func_values) ** 2)
    return np.sqrt(mse)


def plot_graph(data_array, coeffs, funcs, title, save_path=None):
    t_values = np.arange(1, len(data_array) + 1)
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

    funcs = [
        f,
        g,
        h,
        s,
        t,
        lambda x: p(x, 2 * np.pi / 20),
        lambda x: p(x, 2 * np.pi / 40),
        lambda x: p(x, 2 * np.pi / 60),
        lambda x: r(x, 0.005),
    ]

    coeffs = calc_coeffs(data_array, funcs)
    print("Коэффициенты:", *coeffs)

    rmse = calculate_rmse(data_array, coeffs, funcs)
    print(f"Среднеквадратичная ошибка (RMSE): {rmse:.2f}")

    plot_graph(
        data_array,
        coeffs,
        funcs,
        "Обобщённая регрессия для цен акций (расширенная модель)",
        save_path=ASSETS_DIR / "regression_extended.png",
    )


if __name__ == "__main__":
    main()
