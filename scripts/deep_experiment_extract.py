"""Extract detailed tables and comparison info from key papers."""
import re
import os

# Focus on the 3 most relevant papers for our analysis
papers_to_deep_dive = {
    'Su et al. (Islanded MG TSEC)': {
        'path': 'temp/paper_texts/Safe_Reinforcement_Learning-Based_Transient_Stability_Control_for_Islanded_Microgrids_With_Topology_Reconfiguration.txt',
        'sections': [
            ('TABLE I', 'Estimation model comparison'),
            ('TABLE II', 'RL model comparison'),
            ('TABLE III', 'TSEC method comparison'),
            ('Case', 'Case study details'),
        ]
    },
    'Xia et al. (NMG SDRL)': {
        'path': 'temp/paper_texts/Hierarchical_Coordination_of_Networked-Microgrids_Toward_Decentralized_Operation_A_Safe_Deep_Reinforcement_Learning_Method.txt',
        'sections': [
            ('TABLE I', 'System parameters'),
            ('TABLE II', 'DRL settings'),
            ('comparison', 'Comparison methods'),
            ('CASE', 'Case study'),
        ]
    },
    'Barbalho et al. (MG RL Survey)': {
        'path': 'temp/paper_texts/Reinforcement_Learning_Solutions_for_Microgrid_Control_and_Management_A_Survey.txt',
        'sections': [
            ('TABLE', 'Tables with comparisons'),
            ('research gap', 'Research gaps'),
            ('trend', 'Research trends'),
        ]
    },
    'Yu et al. (Power Sys Safe RL)': {
        'path': 'temp/paper_texts/2407.00681v1.txt',
        'sections': [
            ('TABLE', 'Tables'),
            ('challenge', 'Challenges'),
            ('future', 'Future directions'),
            ('frequency', 'Frequency regulation'),
            ('voltage', 'Voltage control'),
            ('energy management', 'Energy management'),
        ]
    },
}

output_file = 'temp/paper_analysis/detailed_experiment_analysis.txt'
os.makedirs('temp/paper_analysis', exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as out:
    for name, info in papers_to_deep_dive.items():
        out.write(f'\n{"="*80}\n')
        out.write(f'PAPER: {name}\n')
        out.write(f'{"="*80}\n')

        with open(info['path'], 'r', encoding='utf-8') as f:
            text = f.read()

        # Find all TABLE references
        lines = text.split('\n')
        table_lines = []
        for i, line in enumerate(lines):
            if 'TABLE' in line.upper() and len(line) < 200:
                context_start = max(0, i-1)
                context_end = min(len(lines), i+15)
                for j in range(context_start, context_end):
                    table_lines.append((j, lines[j]))

        if table_lines:
            out.write('\n--- TABLE REFERENCES ---\n')
            prev_ln = -10
            for ln, content in table_lines:
                if ln - prev_ln > 5:
                    out.write(f'\n  [Table block at line {ln}]\n')
                out.write(f'  L{ln}: {content.strip()[:120]}\n')
                prev_ln = ln

        # Find comparison-related content
        out.write('\n--- COMPARISON METHODS & METRICS ---\n')
        comparison_keywords = ['compare', 'comparison', 'baseline', 'benchmark',
                               'PPO', 'SAC', 'DQN', 'DDPG', 'CPO', 'PCPO', 'RCPO',
                               'TRPO', 'Lagrangian', 'NSGA', 'MPC']
        found_methods = set()
        for kw in comparison_keywords:
            count = text.count(kw)
            if count > 0:
                found_methods.add((kw, count))

        if found_methods:
            out.write('  RL algorithms mentioned:\n')
            for method, count in sorted(found_methods, key=lambda x: -x[1]):
                out.write(f'    {method}: {count} occurrences\n')

        # Find performance metrics
        out.write('\n--- PERFORMANCE METRICS ---\n')
        metric_keywords = ['reward', 'cost', 'MAE', 'accuracy', 'violation',
                           'constraint', 'safety', 'load shedding', 'loss',
                           'voltage', 'frequency', 'SoC', 'convergence',
                           'false alarm', 'miss detection', 'TSI']
        found_metrics = set()
        for kw in metric_keywords:
            count = text.lower().count(kw.lower())
            if count > 0:
                found_metrics.add((kw, count))

        if found_metrics:
            out.write('  Metrics mentioned:\n')
            for metric, count in sorted(found_metrics, key=lambda x: -x[1]):
                out.write(f'    {metric}: {count} occurrences\n')

        # Find experiment setup details
        out.write('\n--- EXPERIMENTAL SETUP ---\n')
        setup_keywords = ['episode', 'step', 'training', 'learning rate',
                          'neural network', 'network structure', 'hidden',
                          'optimizer', 'seed', 'environment']
        for kw in setup_keywords:
            # Find first occurrence with context
            idx = text.lower().find(kw.lower())
            if idx > 0:
                context = text[max(0,idx-50):idx+150].replace('\n', ' ')
                out.write(f'  {kw}: ...{context[:200]}...\n')

        # Future challenges / research gaps
        out.write('\n--- CHALLENGES / FUTURE DIRECTIONS ---\n')
        challenge_patterns = [
            r'challenge.*?:.*?(?=\n\n|\n[A-Z]+\. |\Z)',
            r'future.*?:.*?(?=\n\n|\n[A-Z]+\. |\Z)',
            r'research gap.*?:.*?(?=\n\n|\n[A-Z]+\. |\Z)',
            r'gap.*?:.*?(?=\n\n|\n[A-Z]+\. |\Z)',
        ]
        for pat in challenge_patterns:
            matches = re.findall(pat, text, re.IGNORECASE | re.DOTALL)
            for m in matches[:3]:
                if len(m) > 50:
                    out.write(f'  {m[:300]}\n\n')

print(f"Detailed analysis saved to {output_file}")
