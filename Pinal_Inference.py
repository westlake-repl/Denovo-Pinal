from utils.design_utils import load_pinal, PinalDesign

load_pinal()
input_str="Actin."
res = PinalDesign(desc=input_str, num=10)
print(res)