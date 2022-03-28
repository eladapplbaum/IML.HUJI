from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10, 1, 1000)
    ug = UnivariateGaussian()
    ug.fit(samples)
    print(ug.mu_, ug.var_)


    # Question 2 - Empirically showing sample mean is consistent
    x_axis = []
    y_axis = []
    for i in range(0, samples.size, 10):
        ug.fit(samples[:i + 10])
        x_axis.append(i + 10)
        y_axis.append(abs(10 - ug.mu_))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="markers"))
    fig.update_layout(title="Exception difference vs sample size",
                      xaxis_title="Sample size",
                      yaxis_title="Exception difference")
    fig.show()

    # # Question 3 - Plotting Empirical PDF of fitted model
    x_axis_2 = samples
    ug.fit(samples)
    y_axis_2 = ug.pdf(samples)
    fig_2 = go.Figure()
    fig_2.add_trace(go.Scatter(x=x_axis_2, y=y_axis_2, mode="markers"))
    fig_2.update_layout(
        title="the empirical PDF function under the fitted model",
        xaxis_title="Ordered sample values",
        yaxis_title="PDF function")
    fig_2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov_mat = np.array([[1, 0.2, 0, 0.5],
               [0.2, 2, 0, 0],
               [0, 0, 1, 0],
               [0.5, 0, 0, 1]])
    samples = np.random.multivariate_normal(
        mean= mu,
        cov=cov_mat,
        size=1000)
    mg = MultivariateGaussian()
    mg.fit(samples)
    print(mg.mu_)
    print(mg.cov_)

    # Question 5 - Likelihood evaluation
    mat = np.linspace(-10, 10, 200)
    z_log_likelihood = np.empty(shape=(mat.size, mat.size))
    max_likelihood_argumants = [mat[0], mat[0], mg.log_likelihood(np.array([mat[0], 0, mat[0], 0]), cov_mat, samples)]
    for i in range(mat.size):
        for j in range(mat.size):
            z_log_likelihood[i][j] = mg.log_likelihood(np.array([mat[i], 0, mat[j], 0]), cov_mat, samples)
            if(z_log_likelihood[i][j] > max_likelihood_argumants[2]):
                max_likelihood_argumants = [mat[i], mat[j], z_log_likelihood[i][j]]
    fig_3 = go.Figure()
    fig_3.add_trace(go.Heatmap(x=mat, y=mat ,z=z_log_likelihood))
    fig_3.update_layout(
        title="the log_likelihood of mu=[f1,0,f3,0] as function of f1 and f3",
        xaxis_title="f1 values",
        yaxis_title="f3 values")
    fig_3.show()


    # # Question 6 - Maximum likelihood
    print(max_likelihood_argumants[0],max_likelihood_argumants[1])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
