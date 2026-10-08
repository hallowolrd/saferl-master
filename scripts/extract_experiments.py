"""Extract experiment sections from all papers."""
import re
import os

papers = {
    'Gu et al. (Safe RL Survey)': 'temp/paper_texts/A_Review_of_Safe_Reinforcement_Learning_Methods_Theories_and_Applications.txt',
    'Bian et al. (Fuzzy RL MAS)': 'temp/paper_texts/Fuzzy_Reinforcement_Learning-Based_Safe_Cooperative_Control_for_Nonlinear_Multiagent_Systems.txt',
    'Su et al. (Islanded MG TSEC)': 'temp/paper_texts/Safe_Reinforcement_Learning-Based_Transient_Stability_Control_for_Islanded_Microgrids_With_Topology_Reconfiguration.txt',
    'Xia et al. (NMG SDRL)': 'temp/paper_texts/Hierarchical_Coordination_of_Networked-Microgrids_Toward_Decentralized_Operation_A_Safe_Deep_Reinforcement_Learning_Method.txt',
    'Barbalho et al. (MG RL Survey)': 'temp/paper_texts/Reinforcement_Learning_Solutions_for_Microgrid_Control_and_Management_A_Survey.txt',
    'Yu et al. (Power Sys Safe RL)': 'temp/paper_texts/2407.00681v1.txt',
}

output_file = 'temp/paper_analysis/experiment_sections.txt'
os.makedirs('temp/paper_analysis', exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as out:
    for name, path in papers.items():
        out.write(f'\n{"="*80}\n')
        out.write(f'PAPER: {name}\n')
        out.write(f'{"="*80}\n')

        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Find experiment/case study section
        patterns = [
            r'IV\. CASE STUDY.*?(?=V\. |VI\. |APPENDIX|REFERENCES|\Z)',
            r'IV\. EXPERIMENT.*?(?=V\. |VI\. |APPENDIX|REFERENCES|\Z)',
            r'IV\. SIMULATION.*?(?=V\. |VI\. |APPENDIX|REFERENCES|\Z)',
            r'IV\. RESULTS.*?(?=V\. |VI\. |APPENDIX|REFERENCES|\Z)',
            r'III\. EXPERIMENT.*?(?=IV\. |V\. |REFERENCES|\Z)',
            r'IV\. NUMERICAL.*?(?=V\. |VI\. |REFERENCES|\Z)',
        ]

        found = False
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if match and len(match.group()) > 500:
                result = match.group()
                out.write(f'\n--- EXPERIMENT SECTION ({len(result)} chars) ---\n')
                # Write first 5000 chars
                out.write(result[:5000] + '\n')
                if len(result) > 5000:
                    out.write(f'\n... [TRUNCATED - total {len(result)} chars] ...\n')
                found = True
                break

        if not found:
            # Try to find any table or result-related content
            out.write('\n--- No dedicated experiment section found. Searching for result content... ---\n')
            # Look for table mentions, result descriptions
            table_indicators = ['TABLE', 'Table', 'result', 'comparison', 'baseline', 'benchmark']
            lines = text.split('\n')
            relevant_lines = []
            for i, line in enumerate(lines):
                for ind in table_indicators:
                    if ind in line and len(relevant_lines) < 100:
                        start = max(0, i-2)
                        end = min(len(lines), i+5)
                        for j in range(start, end):
                            if j not in [r[0] for r in relevant_lines]:
                                relevant_lines.append((j, lines[j]))
                        break
            if relevant_lines:
                for ln, content in sorted(relevant_lines, key=lambda x: x[0]):
                    out.write(f'L{ln}: {content}\n')

        # Also extract last few pages for conclusion / future work info
        out.write('\n--- LATER SECTIONS (searching for results/discussion) ---\n')
        # Find pages 8+ which likely contain experiments
        page_matches = list(re.finditer(r'=== PAGE (\d+) ===', text))
        if len(page_matches) >= 6:
            mid_page = page_matches[len(page_matches)//2]
            out.write(text[mid_page.start():mid_page.start()+3000] + '\n')

print(f"Experiment sections saved to {output_file}")
