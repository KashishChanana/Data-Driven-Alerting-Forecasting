import matplotlib.pyplot as plt
import shap


def interpret_model(model, x_train, explainer='Explainer'):
    """
    Displays plots obtained using SHAP for interpret machine learning model i.e, explain its output

    Parameters
    ----------------
    :param model: the trained/fitted model
    :param x_train: the training dataframe for the model
    :param explainer : SHAP parameter for describing model category

    Returns
    ----------------
    None; displays plots
    """

    if explainer == 'Explainer':
        explainer = shap.Explainer(model, x_train)
    elif explainer == 'Tree':
        explainer = shap.TreeExplainer(model, x_train)
    elif explainer == 'Kernel':
        explainer = shap.KernelExplainer(model, x_train)
    elif explainer == 'Deep':
        explainer = shap.DeepExplainer(model, x_train)

    shap_values = explainer(x_train)

    # Uncomment below for summary plot
    # shap.summary_plot(shap_values, x_train, plot_type="bar")

    # summarize the effects of all the features
    shap.plots.beeswarm(shap_values)
    plt.show()

    # shap.plots.bar(shap_values)
    # plt.show()
