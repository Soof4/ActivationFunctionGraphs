from matplotlib.font_manager import FontProperties
import numpy as np
import matplotlib.pyplot as plt
import math

# Plot drawing wrappers
def s_plot(x, y, label, color):
    plt.plot(x, y, label=label, color=color, linewidth=4)

def s_label(x_name, y_name):
    plt.xlabel(x_name, fontsize=16, fontweight='bold')
    plt.ylabel(y_name, fontsize=16, fontweight='bold')

def init_graph():
    x = np.linspace(-6, 6, 500)
    plt.figure(figsize=(8, 6))
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(alpha=0.3)
    return x

def s_legend():
    font_properties = FontProperties(weight='bold', size=16)
    plt.legend(prop=font_properties, loc='lower right')


# Definition of activation functions
def step(t, x):
    return np.where(x >= t, 1, 0)

def relu(x):
    return np.maximum(0, x)

def elu(alpha, x):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Graph show functions
def show_step():
    x = init_graph()

    y_stepn1 = step(-1, x)
    y_step0 = step(0, x)
    y_step1 = step(1, x)
    
    s_plot(x, y_stepn1, 'Step (t=-1)', 'blue')
    s_plot(x, y_step0, 'Step (t=0)', 'red')
    s_plot(x, y_step1, 'Step (t=1)', 'green')
    s_label('x', 'Step(x)')
    s_legend()

    plt.show()

def show_relu():
    x = init_graph()
    y_relu = relu(x)

    s_plot(x, y_relu, 'ReLU', color='blue')
    s_label('x', 'ReLU(x)')
    
    plt.show()

def show_elu():
    x = init_graph()

    y_elu05 = elu(0.5, x)
    y_elu1 = elu(1, x)
    y_elu2 = elu(2, x)

    s_plot(x, y_elu05, 'ELU (α=0.5)', 'blue')
    s_plot(x, y_elu1, 'ELU (α=1)', 'red')
    s_plot(x, y_elu2, 'ELU (α=2)', 'green')
    s_label('x', 'ELU(x)')
    s_legend()
    
    plt.show()

def show_sigmoid():
    x = init_graph()

    y_sigmoid = sigmoid(x)
    s_plot(x, y_sigmoid, 'Sigmoid', 'blue')
    s_label('x', 'Sigmoid(x)')

    plt.show()

def show_tanh():
    x = init_graph()
    
    y_tanh = tanh(x)
    s_plot(x, y_tanh, 'Tanh', 'blue')
    s_label('x', 'Tanh(x)')
    plt.show()

show_step()
show_elu()
show_relu()
show_tanh()
show_sigmoid()
