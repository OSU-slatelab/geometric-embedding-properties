import sys
import os

target_list_path = sys.argv[1]
new_ext = 'bin'

with open(target_list_path) as f:
    target_list = f.readlines()

for i,target in enumerate(target_list):
    target = os.path.abspath(target)
    target = list(target)
    target.remove('\n')
    target = "".join(target)
    extension = target.split('.')[-1]
    new_targ = '.'.join(target.split('.')[:len(target.split('.')) - 1]) + '.' + new_ext
    os.system('mv' + ' ' + target + ' ' + new_targ)
