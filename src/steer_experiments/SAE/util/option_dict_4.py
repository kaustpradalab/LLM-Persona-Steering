def get_option_token(option_type):
    if option_type=="ABCD":
        return ["A", "B", "C", "D"]
    elif option_type=="1234":
        return ["1", "2", "3", "4"]
    else:
        raise ValueError(f"option_type should be 'ABCD' or '1234', but got {option_type}")
    
def get_demographic_pair(prompt):
    if "gender" in prompt:
        return ["female", "male"]
    elif "age" in prompt:
        return ["young", "old"]
    elif "socioeconomic status" in prompt:
        return ["rich", "poor"]
    elif "social ideology" in prompt:
        return ["liberal", "communism"]