import os

datasets = ['flintstones', 'pororo']
models = ['diffstory','story-dalle','storyldm']
model_prefix = ['llm_ref','sd','sd_ft_ref','sd_ft_img','sd_ft_ref','sd_ft_text','sd_ref']
for DATASET in datasets:
    for model in models:
        for prefix in model_prefix:
            if model == 'diffstory':
                filename = DATASET + '_' + prefix
                GEN_DIR = os.path.join('/ibex/ai/home/shenx/.outputs',model,filename)
                print(GEN_DIR)
                if os.path.exists(GEN_DIR):
                    os.system(f'sbatch -o {DATASET}_ours_{prefix}.log\
                            --export=GEN_DIR={GEN_DIR},DATASET={DATASET} eval.sh')
            else:
                break
        if model != 'diffstory':
            GEN_DIR = os.path.join('/ibex/ai/home/shenx/.outputs',model,DATASET)
            os.system(f'sbatch -o {DATASET}_{model}.log\
                    --export=GEN_DIR={GEN_DIR},DATASET={DATASET} eval.sh')
            GEN_DIR = os.path.join('/ibex/ai/home/shenx/.outputs',model,DATASET+'_ref')
            os.system(f'sbatch -o {DATASET}_{model}_ref.log\
                    --export=GEN_DIR={GEN_DIR},DATASET={DATASET} eval.sh')
