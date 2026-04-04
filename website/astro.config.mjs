// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
	integrations: [
		starlight({
			title: 'NeoGraph',
			description: 'Declarative LLM graph compiler. Define typed Nodes, compose into Constructs, compile to LangGraph.',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/KonstantinMirin/neograph' }],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'What is NeoGraph?', slug: 'getting-started/what-is-neograph' },
						{ label: 'Quick Start', slug: 'getting-started/quick-start' },
						{ label: 'Why Not Just LangGraph?', slug: 'getting-started/why-neograph' },
					],
				},
				{
					label: 'Core Concepts',
					items: [
						{ label: 'Vocabulary', slug: 'concepts/vocabulary' },
						{ label: 'Node Modes', slug: 'concepts/node-modes' },
						{ label: 'Modifiers', slug: 'concepts/modifiers' },
						{ label: 'Subgraphs', slug: 'concepts/subgraphs' },
						{ label: 'Observability', slug: 'concepts/observability' },
						{ label: 'LLM Configuration', slug: 'concepts/llm-configuration' },
					],
				},
				{
					label: 'Walkthrough',
					items: [
						{ label: '1. Scripted Pipeline', slug: 'walkthrough/scripted-pipeline' },
						{ label: '2. LLM Produce + Gather', slug: 'walkthrough/produce-and-gather' },
						{ label: '3. Oracle Ensemble', slug: 'walkthrough/oracle-ensemble' },
						{ label: '4. Each Fan-Out', slug: 'walkthrough/each-fanout' },
						{ label: '5. Human-in-the-Loop', slug: 'walkthrough/human-in-the-loop' },
						{ label: '6. Full Pipeline', slug: 'walkthrough/full-pipeline' },
					],
				},
				{
					label: 'vs LangGraph',
					items: [
						{ label: 'Side-by-Side Comparison', slug: 'comparison/overview' },
					],
				},
				{
					label: 'Reference',
					items: [
						{ label: 'API Reference', slug: 'reference/api' },
					],
				},
			],
		}),
	],
});
