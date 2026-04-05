// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
	integrations: [
		starlight({
			title: 'neograph',
			description: 'Write Python. Get a production graph. Declarative LLM graph compiler.',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/KonstantinMirin/neograph' }],
			customCss: ['./src/styles/custom.css'],
			components: {
				SiteTitle: './src/components/SiteTitle.astro',
				Banner: './src/components/Banner.astro',
			},
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
					label: 'The @node API',
					items: [
						{ label: 'Functions as Nodes', slug: 'node-api/functions-as-nodes' },
						{ label: 'Modifiers as Keywords', slug: 'node-api/modifier-kwargs' },
						{ label: 'Non-node Parameters', slug: 'node-api/parameters' },
						{ label: 'Organizing Pipelines', slug: 'node-api/organizing' },
					],
				},
				{
					label: 'ForwardConstruct',
					items: [
						{ label: 'Python Control Flow', slug: 'forward/control-flow' },
						{ label: 'Branching & Loops', slug: 'forward/branching' },
					],
				},
				{
					label: 'Core Concepts',
					items: [
						{ label: 'Node Modes', slug: 'concepts/node-modes' },
						{ label: 'Subgraphs', slug: 'concepts/subgraphs' },
						{ label: 'Observability', slug: 'concepts/observability' },
						{ label: 'LLM Configuration', slug: 'concepts/llm-configuration' },
						{ label: 'Testing', slug: 'concepts/testing' },
					],
				},
				{
					label: 'Runtime Construction',
					items: [
						{ label: 'Programmatic API', slug: 'runtime/programmatic' },
						{ label: 'LLM-Driven Pipelines', slug: 'runtime/llm-driven' },
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
