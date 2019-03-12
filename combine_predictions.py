import os
import shutil

def recombine_predictions (repredictions_dir, output_dir):
    for epoch in os.listdir(repredictions_dir):
        if not epoch.isnumeric():
            continue
        epoch_path = os.path.join(repredictions_dir, epoch, 'obj')
        for train_test_type in os.listdir(epoch_path):
            if train_test_type not in ('train', 'test'):
                continue
            tt_path = os.path.join(epoch_path, train_test_type)
            for file in os.listdir(tt_path):
                name, input_path = file.split('.')[0], os.path.join(tt_path, file)
                base_output_path = os.path.join(output_dir, '%s-%s'%(train_test_type, name))
                if file.endswith('.input.gen.obj'):
                    output_path = os.path.join(base_output_path, 'input.obj')
                elif file.endswith('.output.gen.obj'):
                    output_path = os.path.join(base_output_path, 'epoch-%s.obj'%epoch)
                else:
                    continue
                if not os.path.exists(base_output_path):
                    os.makedirs(base_output_path)
                print("copying %s => %s"%(input_path, output_path))
                shutil.copy(input_path, output_path)

if __name__ == '__main__':
    recombine_predictions('repredicted', 'predictions')
