#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <functional>
#include <stdexcept>

using namespace std;
using namespace std::chrono;

// Linear Algebra Methods

void gaussianElimination(vector<vector<double>>& A, vector<double>& b) {
    int n = A.size();
    vector<double> x(n);

    // Forward Elimination
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            double ratio = A[j][i] / A[i][i];
            for (int k = i; k < n; k++)
                A[j][k] -= ratio * A[i][k];
            b[j] -= ratio * b[i];
        }
    }

    // Back Substitution
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= A[i][j] * x[j];
        x[i] /= A[i][i];
    }
    cout << "Gaussian Elimination Solution:\n";
    for (double val : x)
        cout << fixed << setprecision(6) << val << " ";
    cout << endl;
}

void luDecomposition(const vector<vector<double>>& A, vector<vector<double>>& L, vector<vector<double>>& U) {
    int n = A.size();
    L = vector<vector<double>>(n, vector<double>(n, 0.0));
    U = vector<vector<double>>(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int k = i; k < n; k++) {
            U[i][k] = A[i][k];
            for (int j = 0; j < i; j++)
                U[i][k] -= L[i][j] * U[j][k];
        }
        for (int k = i; k < n; k++) {
            if (i == k)
                L[i][i] = 1;
            else {
                L[k][i] = A[k][i];
                for (int j = 0; j < i; j++)
                    L[k][i] -= L[k][j] * U[j][i];
                L[k][i] /= U[i][i];
            }
        }
    }

    cout << "LU Decomposition L:\n";
    for (const auto& row : L) {
        for (double val : row)
            cout << fixed << setprecision(6) << val << " ";
        cout << endl;

    }

    cout << "LU Decomposition U:\n";
    for (const auto& row : U) {
        for (double val : row)
            cout << fixed << setprecision(6) << val << " ";
        cout << endl;
    }
}

vector<double> jacobiMethod(const vector<vector<double>>& A, const vector<double>& b, double tol = 1e-10, int max_iter = 1000) {
    int n = A.size();
    vector<double> x(n, 0.0), x_new(n, 0.0);

    for (int it = 0; it < max_iter; it++) {
        for (int i = 0; i < n; i++) {
            x_new[i] = b[i];
            for (int j = 0; j < n; j++) {
                if (i != j)
                    x_new[i] -= A[i][j] * x[j];
            }
            x_new[i] /= A[i][i];
        }

        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += abs(x_new[i] - x[i]);
        }
        if (norm < tol)
            break;
        x = x_new;
    }

    cout << "Jacobi Method Solution:\n";
    for (double val : x)
        cout << fixed << setprecision(6) << val << " ";
    cout << endl;

    return x;
}

vector<double> gaussSeidelMethod(const vector<vector<double>>& A, const vector<double>& b, double tol = 1e-10, int max_iter = 1000) {
    int n = A.size();
    vector<double> x(n, 0.0);

    for (int it = 0; it < max_iter; it++) {
        vector<double> x_old = x;
        for (int i = 0; i < n; i++) {
            x[i] = b[i];
            for (int j = 0; j < n; j++) {
                if (i != j)
                    x[i] -= A[i][j] * x[j];
            }
            x[i] /= A[i][i];
        }

        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += abs(x[i] - x_old[i]);
        }
        if (norm < tol)
            break;
    }

    cout << "Gauss-Seidel Method Solution:\n";
    for (double val : x)
        cout << fixed << setprecision(6) << val << " ";
    cout << endl;

    return x;
}

// Non-Linear Methods

double bisectionMethod(function<double(double)> func, double a, double b, double tol = 1e-10, int max_iter = 1000) {
    double c;
    for (int i = 0; i < max_iter; i++) {
        c = (a + b) / 2;
        if (func(c) == 0.0 || (b - a) / 2 < tol)
            break;
        if (func(c) * func(a) < 0)
            b = c;
        else
            a = c;
    }

    return c;
}

double newtonRaphsonMethod(function<double(double)> func, function<double(double)> dfunc, double x0, double tol = 1e-10, int max_iter = 1000) {
    double x = x0;
    for (int i = 0; i < max_iter; i++) {
        double fx = func(x);
        double dfx = dfunc(x);
        if (dfx == 0)
            throw runtime_error("Zero derivative. No solution found.");
        double x_new = x - fx / dfx;
        if (abs(x_new - x) < tol)
            return x_new;
        x = x_new;
    }
    return x;
}

double secantMethod(function<double(double)> func, double x0, double x1, double tol = 1e-10, int max_iter = 1000) {
    double x_prev = x0, x_curr = x1;
    for (int i = 0; i < max_iter; i++) {
        double fx_prev = func(x_prev);
        double fx_curr = func(x_curr);
        double x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev);
        if (abs(x_new - x_curr) < tol)
            return x_new;
        x_prev = x_curr;
        x_curr = x_new;
    }
    return x_curr;
}

double fixedPointIteration(function<double(double)> g, double x0, double tol = 1e-10, int max_iter = 1000) {
    double x = x0;
    for (int i = 0; i < max_iter; i++) {
        double x_new = g(x);
        if (abs(x_new - x) < tol)
            return x_new;
        x = x_new;
    }
    return x;
}

int main() {
    int choice;
    cout << "Select the type of equations:\n";
    cout << "1. Linear Equations\n";
    cout << "2. Non-Linear Equations\n";
    cin >> choice;

    if (choice == 1) {
        // Linear equations
        int n;
        cout << "Enter the number of variables: ";
        cin >> n;

        vector<vector<double>> A(n, vector<double>(n));
        vector<double> b(n);

        cout << "Enter the coefficients matrix A:\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cin >> A[i][j];
            }
        }

        cout << "Enter the constants vector b:\n";
        for (int i = 0; i < n; i++) {
            cin >> b[i];
        }

        int method;
        cout << "Select the numerical method:\n";
        cout << "1. Gaussian Elimination\n";
        cout << "2. LU Decomposition\n";
        cout << "3. Jacobi Method\n";
        cout << "4. Gauss-Seidel Method\n";
        cin >> method;

        switch (method) {
            case 1:
                gaussianElimination(A, b);
                break;
            case 2: {
                vector<vector<double>> L, U;
                luDecomposition(A, L, U);
                break;
            }
            case 3:
                jacobiMethod(A, b);
                break;
            case 4:
                gaussSeidelMethod(A, b);
                break;
            default:
                cout << "Invalid choice.\n";
                break;
        }
    } else if (choice == 2) {
        // Non-linear equations
        int non_linear_choice;
        cout << "Select the non-linear method:\n";
        cout << "1. Bisection Method\n";
        cout << "2. Newton-Raphson Method\n";
        cout << "3. Secant Method\n";
        cout << "4. Fixed Point Iteration\n";
        cin >> non_linear_choice;

        double a, b, x0, x1;

        switch (non_linear_choice) {
            case 1:
                cout << "Enter the function for Bisection Method in the form 'x*x - 2' (use x as variable):\n";
                cout << "Enter the interval [a, b]: ";
                cin >> a >> b;
                cout << "Root: " << bisectionMethod([](double x) { return x*x - 2; }, a, b) << endl;
                break;
            case 2:
                cout << "Enter the function for Newton-Raphson Method in the form 'x*x - 2' (use x as variable):\n";
                cout << "Enter the derivative function in the form '2*x':\n";
                cout << "Enter an initial guess: ";
                cin >> x0;
                cout << "Root: " << newtonRaphsonMethod([](double x) { return x*x - 2; }, [](double x) { return 2*x; }, x0) << endl;
                break;
            case 3:
                cout << "Enter the function for Secant Method in the form 'x*x - 2' (use x as variable):\n";
                cout << "Enter two initial guesses: ";
                cin >> x0 >> x1;
                cout << "Root: " << secantMethod([](double x) { return x*x - 2; }, x0, x1) << endl;
                break;
            case 4:
                cout << "Enter the function for Fixed Point Iteration in the form 'sqrt(x + 2)' (use x as variable):\n";
                cout << "Enter an initial guess: ";
                cin >> x0;
                cout << "Fixed Point: " << fixedPointIteration([](double x) { return sqrt(x + 2); }, x0) << endl;
                break;
            default:
                cout << "Invalid choice.\n";
                break;
        }
    } else {
        cout << "Invalid choice.\n";
    }

    return 0;
}
