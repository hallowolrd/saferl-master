"""Extract key sections from all papers for analysis."""
import re
import os

papers = [
    'temp/paper_texts/A_Review_of_Safe_Reinforcement_Learning_Methods_Theories_and_Applications.txt',
    'temp/paper_texts/Fuzzy_Reinforcement_Learning-Based_Safe_Cooperative_Control_for_Nonlinear_Multiagent_Systems.txt',
    'temp/paper_texts/Safe_Reinforcement_Learning-Based_Transient_Stability_Control_for_Islanded_Microgrids_With_Topology_Reconfiguration.txt',
    'temp/paper_texts/Hierarchical_Coordination_of_Networked-Microgrids_Toward_Decentralized_Operation_A_Safe_Deep_Reinforcement_Learning_Method.txt',
    'temp/paper_texts/Reinforcement_Learning_Solutions_for_Microgrid_Control_and_Management_A_Survey.txt',
    'temp/paper_texts/2407.00681v1.txt',
]

output_file = 'temp/paper_analysis/extracted_sections.txt'
os.makedirs('temp/paper_analysis', exist_ok=True)

with open(output_file, 'w', encoding='utf-8') as out:
    for p in papers:
        name = p.split('/')[-1].replace('.txt','')
        with open(p, 'r', encoding='utf-8') as f:
            text = f.read()

        out.write(f'\n{"="*80}\n')
        out.write(f'PAPER: {name}\n')
        out.write(f'{"="*80}\n')

        # Abstract (first page)
        first_page_end = text.find('=== PAGE 2 ===')
        if first_page_end > 0:
            abstract_text = text[:first_page_end]
            # Find abstract
            abs_match = re.search(r'Abstract.*?I\.\s+INTRODUCTION', abstract_text, re.DOTALL | re.IGNORECASE)
            if abs_match:
                out.write('\n--- ABSTRACT ---\n')
                out.write(abs_match.group()[:1000] + '\n')

        # Find conclusion section
        patterns = [
            r'V\. CONCLUSION.*?(?=VI\. |APPENDIX|REFERENCES|ACKNOWLEDGMENT|\Z)',
            r'CONCLUSION.*?(?=ACKNOWLEDGMENT|REFERENCES|\Z)',
            r'V\. DISCUSSION.*?(?=VI\. |APPENDIX|REFERENCES|\Z)',
        ]

        found = False
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if match and len(match.group()) > 200:
                out.write('\n--- CONCLUSION ---\n')
                out.write(match.group()[:2000] + '\n')
                found = True
                break

        if not found:
            out.write('\n--- END OF PAPER (last 2000 chars) ---\n')
            out.write(text[-2000:] + '\n')

        # Key method section
        method_patterns = [
            r'III\. PROPOSED.*?(?=IV\. |V\. |REFERENCES|\Z)',
            r'III\. METHODOLOGY.*?(?=IV\. |V\. |REFERENCES|\Z)',
            r'III\. METHOD.*?(?=IV\. |V\. |REFERENCES|\Z)',
            r'II\. PROBLEM.*?(?=III\. |IV\. |REFERENCES|\Z)',
        ]

        for pat in method_patterns:
            match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
            if match and len(match.group()) > 300:
                out.write('\n--- METHOD SECTION (first 3000 chars) ---\n')
                out.write(match.group()[:3000] + '\n')
                break

print(f"Extracted sections saved to {output_file}")
