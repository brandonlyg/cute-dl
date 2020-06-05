# coding=utf-8


def display_params(params):
    print("params: ")
    for i in range(len(params)):
        p = params[i]
        print("name:", p.name, " value:", p.value, " gradient:", p.gradient)
