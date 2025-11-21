

def extract_training_log_info(log_path):

    epochs = []
    losses= []
    RMSE_E_per_atom = []
    RMSE_F = []
    with open(log_path, 'r') as file:
        for line in file:
            if 'Epoch' in line:
                epoch = int(line.split('Epoch')[1].split(':')[0].strip())
                epochs.append(epoch)
                loss = float(line.split('loss=')[1].split(',')[0].strip())
                losses.append(loss)
                RMSE_E_per_atom_str = line.split('RMSE_E_per_atom=')[1].split('meV,')[0].strip()
                RMSE_E_per_atom.append(float(RMSE_E_per_atom_str))
                RMSE_F_str = line.split('RMSE_F=')[1].split('meV')[0].strip()
                RMSE_F.append(float(RMSE_F_str))

    data = {'epochs': epochs, 'loss': losses, 'RMSE_E_per_atom': RMSE_E_per_atom, 'RMSE_F': RMSE_F}
    return data


if __name__=='__main__':
    log_path = '/home/hr492/michaelides-share/hr492/Projects/tartine_project/calculations/testing_gen_1_models/naive_model_testing/tartine_0_run-3.log'
    data = extract_training_log_info(log_path)
    print(data)