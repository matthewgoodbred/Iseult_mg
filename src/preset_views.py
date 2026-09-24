'''Bookkeeping for the preset views saved in .iseult_configs.

A preset is a yml file whose general/ConfigName is the label shown in the
Preset Views menu. The file name is the ConfigName with spaces replaced by
underscores, which is what `iseult.py -p NAME` looks up. The menu order is
kept in ORDER_FILE, one file name per line; presets missing from it (e.g.
freshly saved ones) go after the ordered ones, alphabetically.
'''
import os
import yaml

ORDER_FILE = '.preset_order'
# GenMainParamDict falls back on this one, so it cannot be renamed or deleted.
PROTECTED = ('Default.yml',)


def file_name_for(name):
    return name.strip().replace(' ', '_') + '.yml'


def _config_name(path):
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cfg['general']['ConfigName']
    except Exception:
        return None


def read_order(config_dir):
    try:
        with open(os.path.join(config_dir, ORDER_FILE)) as f:
            return [line.strip() for line in f if line.strip()]
    except OSError:
        return []


def save_order(config_dir, file_names):
    with open(os.path.join(config_dir, ORDER_FILE), 'w') as f:
        f.write(''.join(fname + '\n' for fname in file_names))


def list_presets(config_dir):
    '''Return [(ConfigName, file name), ...] in menu order.'''
    found = {}
    for fname in sorted(os.listdir(config_dir)):
        if fname.endswith('.yml'):
            name = _config_name(os.path.join(config_dir, fname))
            if name is not None:
                found[fname] = str(name)
    order = [fname for fname in read_order(config_dir) if fname in found]
    order += [fname for fname in found if fname not in order]
    return [(found[fname], fname) for fname in order]


def rename_preset(config_dir, file_name, new_name):
    '''Change the preset's ConfigName and move it to the matching file name.
    Returns the new file name. Raises ValueError if the rename isn't allowed.'''
    new_name = new_name.strip()
    if not new_name:
        raise ValueError('The name cannot be empty.')
    if file_name in PROTECTED:
        raise ValueError(f'{file_name} is the fallback view and cannot be renamed.')
    new_file = file_name_for(new_name)
    if os.sep in new_file or (os.altsep and os.altsep in new_file):
        raise ValueError('The name cannot contain a path separator.')
    if new_file != file_name and os.path.exists(os.path.join(config_dir, new_file)):
        raise ValueError(f'A preset saved as {new_file} already exists.')

    order = [fname for _, fname in list_presets(config_dir)]
    old_path = os.path.join(config_dir, file_name)
    with open(old_path) as f:
        cfg = yaml.safe_load(f)
    cfg['general']['ConfigName'] = new_name
    with open(os.path.join(config_dir, new_file), 'w') as f:
        yaml.safe_dump(cfg, f)
    if new_file != file_name:
        os.remove(old_path)
        # Keep the renamed preset where it was in the menu
        save_order(config_dir, [new_file if fname == file_name else fname
                                for fname in order])
    return new_file


def delete_preset(config_dir, file_name):
    if file_name in PROTECTED:
        raise ValueError(f'{file_name} is the fallback view and cannot be deleted.')
    os.remove(os.path.join(config_dir, file_name))
    save_order(config_dir, [fname for fname in read_order(config_dir) if fname != file_name])
