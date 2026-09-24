import pytest
import yaml

import preset_views


def make_preset(config_dir, name, extra=None):
    cfg = {'general': {'ConfigName': name}, 'MainParamDict': extra or {'NumOfRows': 2}}
    (config_dir / preset_views.file_name_for(name)).write_text(yaml.safe_dump(cfg))


@pytest.fixture
def config_dir(tmp_path):
    for name in ['Default', 'beta view', 'alpha']:
        make_preset(tmp_path, name)
    (tmp_path / 'notes.yml').write_text('just: text\n')  # not a preset
    return tmp_path


def names(config_dir):
    return [name for name, _ in preset_views.list_presets(config_dir)]


def test_alphabetical_by_file_name_without_order_file(config_dir):
    assert names(config_dir) == ['Default', 'alpha', 'beta view']


def test_saved_order_is_followed_and_new_presets_go_last(config_dir):
    preset_views.save_order(config_dir, ['beta_view.yml', 'gone.yml', 'Default.yml'])
    make_preset(config_dir, 'aaa')
    assert names(config_dir) == ['beta view', 'Default', 'aaa', 'alpha']


def test_rename_moves_file_keeps_contents_and_position(config_dir):
    preset_views.save_order(config_dir, ['beta_view.yml', 'Default.yml', 'alpha.yml'])
    new_file = preset_views.rename_preset(config_dir, 'beta_view.yml', ' my view ')
    assert new_file == 'my_view.yml'
    assert not (config_dir / 'beta_view.yml').exists()
    cfg = yaml.safe_load((config_dir / new_file).read_text())
    assert cfg == {'general': {'ConfigName': 'my view'}, 'MainParamDict': {'NumOfRows': 2}}
    assert names(config_dir) == ['my view', 'Default', 'alpha']


def test_rename_changing_only_label(config_dir):
    # 'beta_view' maps to the same file as 'beta view'
    assert preset_views.rename_preset(config_dir, 'beta_view.yml', 'beta_view') == 'beta_view.yml'
    assert names(config_dir) == ['Default', 'alpha', 'beta_view']


@pytest.mark.parametrize('file_name, new_name', [
    ('alpha.yml', 'beta view'),   # clobbers another preset
    ('alpha.yml', '   '),
    ('alpha.yml', 'sub/dir'),
    ('Default.yml', 'Other'),
])
def test_rename_refused(config_dir, file_name, new_name):
    before = sorted(p.name for p in config_dir.iterdir())
    with pytest.raises(ValueError):
        preset_views.rename_preset(config_dir, file_name, new_name)
    assert sorted(p.name for p in config_dir.iterdir()) == before


def test_delete(config_dir):
    preset_views.save_order(config_dir, ['alpha.yml', 'Default.yml', 'beta_view.yml'])
    preset_views.delete_preset(config_dir, 'alpha.yml')
    assert not (config_dir / 'alpha.yml').exists()
    assert preset_views.read_order(config_dir) == ['Default.yml', 'beta_view.yml']
    with pytest.raises(ValueError):
        preset_views.delete_preset(config_dir, 'Default.yml')
    assert (config_dir / 'Default.yml').exists()
