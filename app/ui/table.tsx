'use client';

import { useMemo, useState } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  flexRender,
  ColumnDef,
} from '@tanstack/react-table';
import { MathJax } from 'better-react-mathjax';
import { ReactNode } from 'react';

type Row = {
  attention: string;
  precision: string;
  depth: string;
  width: string;
  intermediateStep: string;
  logic: ReactNode;
  automata: ReactNode;
  algebra: ReactNode;
  circuitComplexity: ReactNode;
};

const rawData: Row[] = [
  {
    attention: 'Soft',
    precision: 'Constant',
    depth: 'Constant',
    width: 'Constant',
    intermediateStep: 'None',
    logic: <MathJax inline>{"$=\\pfo$"}</MathJax>,
    automata: <MathJax inline>{"$=$ partially ordered DFA"}</MathJax>,
    algebra: <MathJax inline>{"$\\mathcal{R}$-trivial"}</MathJax>,
    circuitComplexity: <MathJax inline>{"$\\subseteq \\ac$"}</MathJax>,
  },
  {
    attention: 'Average',
    precision: 'Constant',
    depth: 'Constant',
    width: 'Constant',
    intermediateStep: 'None',
    logic: <MathJax inline>{"$=\\pfo$"}</MathJax>,
    automata: <MathJax inline>{"$=$ partially ordered DFA"}</MathJax>,
    algebra: <MathJax inline>{"$\\mathcal{R}$-trivial"}</MathJax>,
    circuitComplexity: <MathJax inline>{"$\\subseteq \\ac$"}</MathJax>,
  },
  {
    attention: 'Leftmost',
    precision: 'Constant',
    depth: 'Constant',
    width: 'Constant',
    intermediateStep: 'None',
    logic: <MathJax inline>{"$=\\pfo$"}</MathJax>,
    automata: <MathJax inline>{"$=$ partially ordered DFA"}</MathJax>,
    algebra: <MathJax inline>{"$=\\mathcal{R}$-trivial"}</MathJax>,
    circuitComplexity: <MathJax inline>{"$\\subseteq \\ac$"}</MathJax>,
  },
  {
    attention: 'Rightmost',
    precision: 'Constant',
    depth: 'Constant',
    width: 'Constant',
    intermediateStep: 'None',
    logic: <MathJax inline>{"$=\\fo$"}</MathJax>,
    automata: <MathJax inline>{"$=$ counter-free DFA"}</MathJax>,
    algebra: <MathJax inline>{"$=$ aperiodic"}</MathJax>,
    circuitComplexity: <MathJax inline>{"$\\subseteq \\ac$"}</MathJax>,
  },
  {
    attention: 'Soft',
    precision: 'Logarithmic',
    depth: 'Constant',
    width: 'Linear',
    intermediateStep: 'None',
    logic: <MathJax inline>{"$\\supseteq (\\textbf{FO+MOD})[<]$"}</MathJax>,
    automata: <MathJax inline>{"$\\supseteq$ solvable"}</MathJax>,
    algebra: <MathJax inline>{"$\\supseteq$ solvable"}</MathJax>,
    circuitComplexity: <MathJax inline>{"$\\supseteq \\text{DLOGTIME-uniform} \\acc$"}</MathJax>,
},
{
attention: 'Soft',
precision: 'Logarithmic',
depth: 'Logarithmic',
width: 'Constant',
intermediateStep: 'None',
logic: <MathJax inline>{"$\\supseteq \\som$"}</MathJax>,
automata: <MathJax inline>{"$\\supseteq$ DFA"}</MathJax>,
algebra: <MathJax inline>{"$\\supseteq$ Finite"}</MathJax>,
circuitComplexity: <MathJax inline>{"$\\supseteq \\tc, \\subseteq\\nc$"}</MathJax>,
},
{
attention: 'Soft',
precision: 'Logarithmic',
depth: 'Constant',
width: 'Constant',
intermediateStep: 'None',
logic: <MathJax inline>{"$\\supseteq \\fo$"}</MathJax>,
automata: <MathJax inline>{"$\\supseteq$ counter-free DFA"}</MathJax>,
algebra: <MathJax inline>{"$\\supseteq$ aperiodic"}</MathJax>,
circuitComplexity: <MathJax inline>{"$\\supseteq \\ac$"}</MathJax> ,
}
];

const filterKeys = [
  'attention',
  'precision',
  'depth',
  'width',
  'intermediateStep',
] as const;

type FilterKey = typeof filterKeys[number];

function formatKeyLabel(key: string) {
  // Capitalize first letter and insert spaces before camelCase capitals
  return key.charAt(0).toUpperCase() + key.slice(1).replace(/([A-Z])/g, ' $1');
}

export default function ExpressivityTable() {
  const [filters, setFilters] = useState<Record<FilterKey, string>>({
    attention: '',
    precision: '',
    depth: '',
    width: '',
    intermediateStep: '',
  });

  const data = useMemo(() => {
    return rawData.filter((row) =>
      filterKeys.every((key) =>
        filters[key] ? row[key] === filters[key] : true
      )
    );
  }, [filters]);

  const filterOptions = useMemo(() => {
    const options: Record<FilterKey, string[]> = {
      attention: [],
      precision: [],
      depth: [],
      width: [],
      intermediateStep: [],
    };
    filterKeys.forEach((key) => {
      options[key] = Array.from(new Set(rawData.map((row) => row[key])));
    });
    return options;
  }, []);

  const columns: ColumnDef<Row>[] = useMemo(
    () => [
      {
        header: 'Assumptions',
        columns: [
          { header: 'Attention', accessorKey: 'attention' },
          { header: 'Precision', accessorKey: 'precision' },
          { header: 'Depth', accessorKey: 'depth' },
          { header: 'Width', accessorKey: 'width' },
          { header: 'Intermediate Step', accessorKey: 'intermediateStep' },
        ],
      },
      {
        header: 'Formal Analysis',
        columns: [
          { header: 'Logic', accessorKey: 'logic', cell: ({ getValue }) => getValue(), },
          { header: 'Automata', accessorKey: 'automata', cell: ({ getValue }) => getValue(), },
          { header: 'Algebra', accessorKey: 'algebra', cell: ({ getValue }) => getValue(), },
          { header: 'Circuit Complexity', accessorKey: 'circuitComplexity', cell: ({ getValue }) => getValue(), },
        ],
      },
    ],
    []
  );

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <div className="flex flex-col gap-6">
      {/* Filters */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-4">
        {filterKeys.map((key) => (
          <select
            key={key}
            value={filters[key]}
            onChange={(e) =>
              setFilters((prev) => ({
                ...prev,
                [key]: e.target.value,
              }))
            }
            className="border rounded px-2 py-1 bg-white"
          >
            <option value="">All {formatKeyLabel(key)}</option>
            {filterOptions[key].map((opt) => (
              <option key={opt} value={opt}>
                {opt}
              </option>
            ))}
          </select>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-auto">
        <table className="min-w-full border border-gray-300 text-sm">
          <thead>
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    colSpan={header.colSpan}
                    className={`border px-3 py-2 bg-gray-100 text-left ${
                      header.colSpan > 1 ? 'text-center font-semibold' : ''
                    }`}
                  >
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext()
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="border px-3 py-2">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
