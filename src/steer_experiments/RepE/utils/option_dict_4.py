def get_option_token(option_type):
    if option_type=="ABCD":
        return ["A", "B", "C", "D"]
    elif option_type=="1234":
        return ["1", "2", "3", "4"]
    else:
        raise ValueError(f"option_type should be 'ABCD' or '1234', but got {option_type}")