import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

quality_range = np.arange(0, 11, 1)
service_range = np.arange(0, 11, 1)
tip_range = np.arange(0, 26, 1)


quality = ctrl.Antecedent(quality_range, 'quality')
service = ctrl.Antecedent(service_range, 'service')
tip = ctrl.Consequent(tip_range, 'tip')

quality['poor'] = fuzz.trimf(quality_range, [0, 0, 5])
quality['average'] = fuzz.trimf(quality_range, [0, 5, 10])
quality['good'] = fuzz.trimf(quality_range, [5, 10, 10])


service['poor'] = fuzz.trimf(service_range, [0, 0, 5])
service['average'] = fuzz.trimf(service_range, [0, 5, 10])
service['good'] = fuzz.trimf(service_range, [5, 10, 10])


tip['low'] = fuzz.trimf(tip_range, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip_range, [0, 13, 25])
tip['high'] = fuzz.trimf(tip_range, [13, 25, 25])


rule1 = ctrl.Rule(quality['poor'] | service['poor'], tip['low'])
rule2 = ctrl.Rule(service['average'], tip['medium'])
rule3 = ctrl.Rule(service['good'] | quality['good'], tip['high'])


tipping_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])


tip_ctrl = ctrl.ControlSystemSimulation(tipping_ctrl)


n_samples = 10
quality_vals = np.random.uniform(0, 10, n_samples)
service_vals = np.random.uniform(0, 10, n_samples)

tip = np.zeros(n_samples)
for i in range(n_samples):
    tip_ctrl.input['quality'] = quality_vals[i]
    tip_ctrl.input['service'] = service_vals[i]
    tip_ctrl.compute()
    tip[i] = tip_ctrl.output['tip']
    
print(tip)