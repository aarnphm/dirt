import inspect
from cattrs.disambiguators import create_uniq_field_dis_func
from bentoml._internal.bento.build_config import (
    BentoBuildConfig,
    CondaOptions,
)

pgetsource = lambda x: print(inspect.getsource(x))
pdir = lambda x: print(dir(x))
p = print

func = create_uniq_field_dis_func(CondaOptions, BentoBuildConfig)
res = func({"dependencies": ["a", "b"]})
p(res)
# pdir(res.dependencies)
# pgetsource(func)
