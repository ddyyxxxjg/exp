import numpy as np
import pandas as pd
import math
import ast
import os

def process_data_file(csv_file_path):
    data = pd.read_csv(csv_file_path)
    data = data.drop(0)
    data = data.dropna(how='any', subset=['rotated_move_mouse.x','number'])
    
    def string_to_float_list(string_data):
        try:
            float_list = ast.literal_eval(string_data)
            if all(isinstance(item, float) for item in float_list):
                return float_list
            else:
                print("Not all elements are floats.")
                return None
        except ValueError as e:
            print("Conversion failed for element:", string_data)
            print("Error:", e)
            return None

    def process_column_to_float_list(column):
        result = []
        for string_data in column:
            float_list = string_to_float_list(string_data)
            if float_list is not None:
                result.append(float_list)
        return result

    rotated_move_mouse_x_list = process_column_to_float_list(data['rotated_move_mouse.x'])
    rotated_move_mouse_y_list = process_column_to_float_list(data['rotated_move_mouse.y'])
    rotated_move_mouse_time_list = process_column_to_float_list(data['rotated_move_mouse.time'])

    data['rotated_move_mouse.x'] = rotated_move_mouse_x_list
    data['rotated_move_mouse.y'] = rotated_move_mouse_y_list
    data['rotated_move_mouse.time'] = rotated_move_mouse_time_list

    target_dict = {}
    original_keys = [10, 20, 30, 40, 50, 60]
    for original_key in original_keys:
        original_data = data.loc[data['number'] == original_key]
        original_values = {
            'number': original_key,
            'target_x': original_data['target_x'].iloc[0],
            'target_y': original_data['target_y'].iloc[0],
            'trials_data': []
        }

        for index, row in original_data.iterrows():
            trial_data = {
                'trials.thisTrialN': row['trials.thisTrialN'],
                'rotated_move_mouse.x': row['rotated_move_mouse.x'],
                'rotated_move_mouse.y': row['rotated_move_mouse.y'],
                'rotated_move_mouse.time': row['rotated_move_mouse.time'],
                'move.started': row['move.started'],
                'rotated_target_circle_2.started': row['rotated_target_circle_2.started']
            }
            original_values['trials_data'].append(trial_data)

        generalization_dict = {}
        generalization_keys = {
            10: [11, 12, 13, 14],
            30: [31, 32, 33, 34],
            50: [51, 52, 53, 54]
        }

        if original_key in [20, 40, 60]:
            generalization_dict = {}
        else:
            for key in generalization_keys.get(original_key, []):
                generalization_data = data.loc[data['number'] == key]

                trials_data = []
                for index, row in generalization_data.iterrows():
                    trial_data = {
                        'trials.thisTrialN': row['trials.thisTrialN'],
                        'rotated_move_mouse.x': row['rotated_move_mouse.x'],
                        'rotated_move_mouse.y': row['rotated_move_mouse.y'],
                        'rotated_move_mouse.time': row['rotated_move_mouse.time'],
                        'move.started': row['move.started'],
                        'rotated_target_circle_2.started': row['rotated_target_circle_2.started']
                    }
                    trials_data.append(trial_data)

                generalization_dict[key] = {
                    'number': key,
                    'target_x': generalization_data['target_x'].iloc[0],
                    'target_y': generalization_data['target_y'].iloc[0],
                    'trials_data': trials_data
                }

        target_dict[original_key] = {
            'original_values': original_values,
            'generalization_dict': generalization_dict
        }

    def append_distances(target_dict):
        for original_key, content in target_dict.items():
            original_values = content['original_values']

            for trial in original_values['trials_data']:
                x_list = trial['rotated_move_mouse.x']
                y_list = trial['rotated_move_mouse.y']
                trial['distances'] = [np.sqrt(x**2 + y**2) for x, y in zip(x_list, y_list)]
        
            for gen_key, gen_values in content['generalization_dict'].items():
                for trial in gen_values['trials_data']:
                    x_list = trial['rotated_move_mouse.x']
                    y_list = trial['rotated_move_mouse.y']
                    trial['distances'] = [np.sqrt(x**2 + y**2) for x, y in zip(x_list, y_list)]

    def append_velocities(target_dict):
        for original_key, content in target_dict.items():
            original_values = content['original_values']

            for trial in original_values['trials_data']:
                x_list = trial['rotated_move_mouse.x']
                y_list = trial['rotated_move_mouse.y']
                time_list = trial['rotated_move_mouse.time']
                distances = [np.sqrt(x**2 + y**2) for x, y in zip(x_list, y_list)]
                velocities = [(distances[i+1] - distances[i]) / (time_list[i+1] - time_list[i]) for i in range(len(time_list) - 1)]
                trial['distances'] = distances  
                trial['velocities'] = velocities
        
            for gen_key, gen_values in content['generalization_dict'].items():
                for trial in gen_values['trials_data']:
                    x_list = trial['rotated_move_mouse.x']
                    y_list = trial['rotated_move_mouse.y']
                    time_list = trial['rotated_move_mouse.time']
                    distances = [np.sqrt(x**2 + y**2) for x, y in zip(x_list, y_list)]
                    velocities = [(distances[i+1] - distances[i]) / (time_list[i+1] - time_list[i]) for i in range(len(time_list) - 1)]
                    trial['distances'] = distances  
                    trial['velocities'] = velocities

    def append_whole_movement_time(target_dict):
        for original_key, data_dict in target_dict.items():
            original_values = data_dict['original_values']
            for trial in original_values['trials_data']:
                time = trial['rotated_move_mouse.time']
                trial['total_motion_time'] = time[-1] - time[0] if time else None
        
            for gen_key, gen_values in data_dict['generalization_dict'].items():
                for trial in gen_values['trials_data']:
                    time = trial['rotated_move_mouse.time']
                    trial['total_motion_time'] = time[-1] - time[0] if time else None

    def append_movement_start_time(target_dict, threshold_factor=0.1):
        def find_movement_start_time(velocities, times, threshold_factor):
            max_velocity = max(velocities)
            threshold = threshold_factor * max_velocity  
            for i, vel in enumerate(velocities):
                if vel > threshold:
                    return times[i]  
            return None

        for original_key, data_dict in target_dict.items():
            original_values = data_dict['original_values']
            for trial in original_values['trials_data']:
                velocities = trial['velocities']  
                times = trial['rotated_move_mouse.time']
                trial['movement_start_time'] = find_movement_start_time(velocities, times, threshold_factor)
        
            for gen_key, gen_values in data_dict['generalization_dict'].items():
                for trial in gen_values['trials_data']:
                    velocities = trial['velocities']  
                    times = trial['rotated_move_mouse.time']
                    trial['movement_start_time'] = find_movement_start_time(velocities, times, threshold_factor)

    def append_closest_to_target_info(target_dict, target_distance=8.0):
        for original_key, content in target_dict.items():
            original_values = content['original_values']
            generalization_dict = content['generalization_dict']

            for trial in list(original_values['trials_data']):  
                distances = trial['distances']
                x_coords = trial['rotated_move_mouse.x']
                y_coords = trial['rotated_move_mouse.y']
                times = trial['rotated_move_mouse.time']

                closest_info = find_closest_info(distances, x_coords, y_coords, times, target_distance)
            
                if all(value is not None for value in closest_info.values()):
                    trial.update(closest_info)
                else:
                    original_values['trials_data'].remove(trial)

            for gen_key, gen_values in generalization_dict.items():
                for trial in list(gen_values['trials_data']):
                    distances = trial['distances']
                    x_coords = trial['rotated_move_mouse.x']
                    y_coords = trial['rotated_move_mouse.y']
                    times = trial['rotated_move_mouse.time']

                    closest_info = find_closest_info(distances, x_coords, y_coords, times, target_distance)
                    if all(value is not None for value in closest_info.values()):
                        trial.update(closest_info)
                    else:
                        gen_values['trials_data'].remove(trial)

    def find_closest_info(distances, x_coords, y_coords, times, target_distance):
        min_diff = float('inf')
        closest_x = None
        closest_y = None
        closest_distance = None
        closest_time = None

        for distance, x, y, time in zip(distances, x_coords, y_coords, times):
            if distance >= target_distance:
                diff = abs(distance - target_distance)
                if diff < min_diff:
                    min_diff = diff
                    closest_x = x
                    closest_y = y
                    closest_distance = distance
                    closest_time = time

        return {
            'closest_x': closest_x,
            'closest_y': closest_y,
            'closest_distance': closest_distance,
            'closest_time': closest_time
        }

    def append_relative_angle(target_dict):
        for original_key, content in target_dict.items():
            original_values = content['original_values']
            generalization_dict = content['generalization_dict']

            for trial in original_values['trials_data']:
                target_x = original_values['target_x']
                target_y = original_values['target_y']
                closest_x = trial['closest_x']
                closest_y = trial['closest_y']
                relative_angle = calculate_single_relative_angle((target_x, target_y), (closest_x, closest_y))
                trial['relative_angle'] = relative_angle
                
            for gen_key, gen_values in generalization_dict.items():
                for trial in gen_values['trials_data']:
                    target_x = gen_values['target_x']
                    target_y = gen_values['target_y']
                    closest_x = trial['closest_x']
                    closest_y = trial['closest_y']
                    relative_angle = calculate_single_relative_angle((target_x, target_y), (closest_x, closest_y))
                    trial['relative_angle'] = relative_angle

    def calculate_single_relative_angle(target_position, closest_position):
        target_r, target_theta = polar_coordinates(target_position)
        closest_r, closest_theta = polar_coordinates(closest_position)

        relative_angle = (closest_theta - target_theta) % 360  

        if relative_angle > 180:
            relative_angle -= 360
        elif relative_angle < -180:
            relative_angle += 360
        return relative_angle

    def polar_coordinates(cartesian):
        x, y = cartesian
        r = math.sqrt(x**2 + y**2)  
        theta = math.degrees(math.atan2(y, x))  
        return r, theta

    append_distances(target_dict)
    append_velocities(target_dict)
    append_whole_movement_time(target_dict)
    append_movement_start_time(target_dict)
    append_closest_to_target_info(target_dict)
    append_relative_angle(target_dict)

    rows = []
    for original_key, content in target_dict.items():
        original_values = content['original_values']
        generalization_dict = content['generalization_dict']

        for trial in original_values['trials_data']:
            row = {
                'type': 'original',
                'number': original_values['number'],
                'target_x': original_values['target_x'],
                'target_y': original_values['target_y'],
                **trial  
            }
            rows.append(row)

        for gen_key, gen_values in generalization_dict.items():
            for trial in gen_values['trials_data']:
                row = {
                    'type': 'generalization',
                    'number': gen_values['number'],
                    'target_x': gen_values['target_x'],
                    'target_y': gen_values['target_y'],
                    **trial  
                }
                rows.append(row)

    df = pd.DataFrame(rows)
    new_column_names = {
        'type': 'type',
        'number': 'target_number',
        'target_x': 'target_x',
        'target_y': 'target_y',
        'trials.thisTrialN': 'trial_index',
        'rotated_move_mouse.x': 'mouse_x',
        'rotated_move_mouse.y': 'mouse_y',
        'rotated_move_mouse.time': 'mouse_time',
        'move.started': 'move_routine_started',
        'rotated_target_circle_2.started': 'target_appearance',
        'distances': 'distances',
        'velocities': 'velocities',
        'total_motion_time':'whole_movement_time',
        'movement_start_time':'start_times',
        'closest_x': 'closest_x',
        'closest_y': 'closest_y',
        'closest_distance': 'closest_distance',
        'closest_time': 'end_times',
        'relative_angle':'relative_angle'}

    df.rename(columns=new_column_names, inplace=True)

    output_folder = "C:/Users/Ying/Desktop/exp_skill_adaptation/Vscode/Data_analysis/000"
    output_file_path = os.path.join(output_folder, f'processed_data_{os.path.basename(csv_file_path)[:3]}.csv')
    df.to_csv(output_file_path, index=False)
    print("Data processing completed. Output saved to:", output_file_path)

