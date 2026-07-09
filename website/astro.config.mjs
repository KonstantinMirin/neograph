// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import starlightClientMermaid from '@pasqal-io/starlight-client-mermaid';
import remarkApi from './plugins/remark-api.mjs';

export default defineConfig({
	// remarkApi validates + autolinks backticked API-symbol references against the
	// introspection-generated manifest (verifiable-docs Stage B). A dotted `Type.member`
	// ref to a fielded type with a missing member FAILS the build.
	markdown: {
		remarkPlugins: [remarkApi],
	},
	integrations: [
		starlight({
			plugins: [starlightClientMermaid()],
			title: 'neograph',
			description: 'Build production AI agents in Python. Functions are nodes, parameter names are edges. The graph assembles itself.',
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
						{ label: 'Sync & Async Execution', slug: 'concepts/async-execution' },
						{ label: 'Checkpoint Resume', slug: 'concepts/checkpoint-resume' },
						{ label: 'Observability', slug: 'concepts/observability' },
						{ label: 'LLM Configuration', slug: 'concepts/llm-configuration' },
						{ label: 'Each x Oracle Fusion', slug: 'concepts/each-oracle-fusion' },
						{ label: 'Prompt Compiler', slug: 'concepts/prompt-compiler' },
						{ label: 'Migrating your prompt compiler', slug: 'concepts/migrating-prompt-compilers' },
						{ label: 'Evaluating Prompts', slug: 'concepts/evaluating-prompts' },
						{ label: 'Input Renderers', slug: 'concepts/renderers' },
						{ label: 'Retry Semantics', slug: 'concepts/retry-semantics' },
						{ label: 'Pipeline Spec Format', slug: 'concepts/spec-format' },
					{ label: 'Testing', slug: 'concepts/testing' },
					{ label: 'Pipeline Validation (neograph check)', slug: 'concepts/check-cli' },
					{ label: 'Static Linting (lint)', slug: 'concepts/lint' },
					{ label: 'Graph Visualization', slug: 'concepts/visualize' },
					{ label: 'Developer Mode', slug: 'concepts/dev-mode' },
					],
				},
				{
					label: 'MCP',
					items: [
						{ label: 'MCP Integration', slug: 'concepts/mcp-integration' },
						{ label: 'Resource Hydration', slug: 'concepts/resource-hydration' },
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
						{ label: '2. LLM Think + Agent', slug: 'walkthrough/produce-and-gather' },
						{ label: '3. Oracle Ensemble', slug: 'walkthrough/oracle-ensemble' },
						{ label: '4. Each Fan-Out', slug: 'walkthrough/each-fanout' },
						{ label: '5. Human-in-the-Loop', slug: 'walkthrough/human-in-the-loop' },
						{ label: '6. Full Pipeline', slug: 'walkthrough/full-pipeline' },
						{ label: '7. Multimodal Vision', slug: 'walkthrough/multimodal-vision' },
						{ label: '8. MCP Client', slug: 'walkthrough/mcp-client' },
						{ label: '9. MCP Resources', slug: 'walkthrough/mcp-resources' },
						{ label: 'More Examples', link: 'https://github.com/KonstantinMirin/neograph/tree/main/examples' },
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
