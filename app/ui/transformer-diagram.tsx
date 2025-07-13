'use client';
import React, { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { MathJax } from 'better-react-mathjax';

// DAG definition (unchanged)
type NodeKey =
  | "embedding"
  | "query"
  | "key"
  | "value"
  | "dot"
  | "argmax"
  | "leftmost"
  | "wsum"
  | "proj"
  | "addnorm1"
  | "ff_linear1"
  | "ff_gelu"
  | "ff_linear2"
  | "addnorm2"
  | "linear";

const DAG: Record<NodeKey, { parents: NodeKey[]; children: NodeKey[] }> = {
  embedding: { parents: [], children: ["query", "key", "value"] },
  query: { parents: ["embedding"], children: ["dot"] },
  key: { parents: ["embedding"], children: ["dot"] },
  value: { parents: ["embedding"], children: ["wsum"] },
  dot: { parents: ["query", "key"], children: ["argmax"] },
  argmax: { parents: ["dot"], children: ["leftmost"] },
  leftmost: { parents: ["argmax"], children: ["wsum"] },
  wsum: { parents: ["leftmost", "value"], children: ["proj"] },
  proj: { parents: ["wsum"], children: ["addnorm1"] },
  addnorm1: { parents: ["proj"], children: ["ff_linear1"] },
  ff_linear1: { parents: ["addnorm1"], children: ["ff_gelu"] },
  ff_gelu: { parents: ["ff_linear1"], children: ["ff_linear2"] },
  ff_linear2: { parents: ["ff_gelu"], children: ["addnorm2"] },
  addnorm2: { parents: ["ff_linear2"], children: ["linear"] },
  linear: { parents: ["addnorm2"], children: [] },
};

const nodeDisplay: Record<NodeKey, { label: string; color: string }> = {
  embedding: { label: "Embedding", color: "bg-blue-700" },
  query: { label: "Query projection", color: "bg-amber-600" },
  key: { label: "Key projection", color: "bg-amber-600" },
  value: { label: "Value projection", color: "bg-amber-600" },
  dot: { label: "Scaled Dot-Product", color: "bg-amber-600" },
  argmax: { label: "Argmax", color: "bg-amber-600" },
  leftmost: { label: "Leftmost", color: "bg-amber-600" },
  wsum: { label: "Weighted sum", color: "bg-amber-600" },
  proj: { label: "Projection", color: "bg-amber-600" },
  addnorm1: { label: "Add", color: "bg-slate-600" },
  ff_linear1: { label: "Linear", color: "bg-lime-600" },
  ff_gelu: { label: "GELU", color: "bg-lime-600" },
  ff_linear2: { label: "Linear", color: "bg-lime-600" },
  addnorm2: { label: "Add", color: "bg-slate-600" },
  linear: { label: "Linear", color: "bg-rose-600" },
};

// displayOrder remains unchanged
const displayOrder: NodeKey[] = [
  "embedding",
  "query", "key", "value",
  "dot", "argmax", "leftmost", "wsum", "proj",
  "addnorm1",
  "ff_linear1", "ff_gelu", "ff_linear2",
  "addnorm2",
  "linear",
];

// *** Responsive TokenBox: font size and box size decrease on small screens
const TokenBox = ({ token }: { token: string }) => (
  <div
    className={`
      border rounded-xl shadow-sm bg-white text-center
      flex items-center justify-center
      transition-all
      w-6 h-6 text-[7px]
      xs:w-7 xs:h-7 xs:text-[8px]
      sm:w-8 sm:h-8 sm:text-[9px]
      md:w-9 md:h-9 md:text-[10px]
      lg:w-10 lg:h-10 lg:text-[11px]
      max-w-[2rem] max-h-[2rem]
      break-all
    `}
    style={{
      minWidth: '1.5rem',
      minHeight: '1.5rem',
    }}
  >
    {token}
  </div>
);


const TokenMatrix = ({ tokens }: { tokens: string[][] }) => (
  <div className="flex flex-col gap-[2px] my-2">
    {tokens.map((row, i) => (
      <div className="flex gap-[2px] justify-center" key={i}>
        {row.map((t, j) => (
          <TokenBox key={j} token={t || ""} />
        ))}
      </div>
    ))}
  </div>
);

// LayerBox and SubLayerBox unchanged except for font responsiveness if needed
const LayerBox = ({
  color,
  label,
  onPrev,
  onNext,
  children,
}: {
  color: string;
  label: string;
  onPrev?: () => void;
  onNext?: () => void;
  children?: React.ReactNode;
}) => (
  <div
    className={`relative rounded-2xl shadow-md px-3 sm:px-4 py-1 sm:py-1.5 w-full max-w-4xl mx-auto ${color} bg-opacity-90
                flex flex-col items-center text-center`}
  >
    {onPrev && (
      <button
        className="absolute left-0 top-1/2 -translate-y-1/2 ml-2 text-white text-xs bg-white/20 px-2 py-1 rounded hover:bg-white/30"
        onClick={onPrev}
        aria-label="Deactivate and hide this layer"
      >
        ▲
      </button>
    )}
    {onNext && (
      <button
        className="absolute right-0 top-1/2 -translate-y-1/2 mr-2 text-white text-xs bg-white/20 px-2 py-1 rounded hover:bg-white/30"
        onClick={onNext}
        aria-label="Activate and show this layer"
      >
        ▼
      </button>
    )}

    {/* Remove margin below */}
    <div className="text-sm sm:text-base font-semibold text-white leading-tight">
      {label}
    </div>

    {children && (
      <div className="mt-0.5">{children}</div>
    )}
  </div>
);

const SubLayerBox = ({
  color,
  label,
  onPrev,
  onNext,
  children,
}: {
  color: string;
  label: string;
  onPrev?: () => void;
  onNext?: () => void;
  children?: React.ReactNode;
}) => (
  <div
    className={`relative rounded-xl shadow p-2 flex flex-col items-center justify-center w-full ${color} bg-opacity-80`}
  >
    {onPrev && (
      <button
        className="absolute left-0 top-1/2 -translate-y-1/2 ml-2 text-white text-[10px] bg-white/20 px-2 py-1 rounded hover:bg-white/30"
        onClick={onPrev}
        aria-label="Deactivate and hide this sublayer"
      >
        ▲
      </button>
    )}
    {onNext && (
      <button
        className="absolute right-0 top-1/2 -translate-y-1/2 mr-2 text-white text-[10px] bg-white/20 px-2 py-1 rounded hover:bg-white/30"
        onClick={onNext}
        aria-label="Activate and show this sublayer"
      >
        ▼
      </button>
    )}

    <span className="text-xs font-semibold text-white leading-tight">
      {label}
    </span>

    {children && (
      <div className="mt-0.5">
        {children}
      </div>
    )}
  </div>
);

const makeMatrix = (rows: number, cols: number, prefix: string) =>
  Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => prefix ? `${prefix}${i + 1}${j + 1}` : "")
  );

export interface TransformerDiagramProps {
  inputTokens: string[];
  vocabulary: Record<string, number>;
  nodeLogics: Record<NodeKey, React.ReactNode>;
  outputs: Record<NodeKey, string[][]>;
}

export default function TransformerDiagram({
  inputTokens,
  vocabulary,
  nodeLogics,
  outputs,
}: TransformerDiagramProps) {
  const [nodeActive, setNodeActive] = useState<Record<NodeKey, boolean>>({
    embedding: false,
    query: false,
    key: false,
    value: false,
    dot: false,
    argmax: false,
    leftmost: false,
    wsum: false,
    proj: false,
    addnorm1: false,
    ff_linear1: false,
    ff_gelu: false,
    ff_linear2: false,
    addnorm2: false,
    linear: false
  });

  const outputsToDisplay = useMemo(() => {
    // for each key in outputs, if that node is active, show the real matrix;
    // otherwise, show a blank matrix of same shape.
    const blankify = (matrix: string[][]) =>
      matrix.map(row => row.map(() => "")); // blank with same shape

    const result: typeof outputs = {} as any;
    for (const k in outputs) {
      result[k as keyof typeof outputs] = nodeActive[k as keyof typeof nodeActive]
        ? outputs[k as keyof typeof outputs]
        : blankify(outputs[k as keyof typeof outputs]);
    }
    return result;
  }, [nodeActive, outputs]);

  const [latestActiveKey, setLatestActiveKey] = useState<NodeKey | null>(null);
  const logicRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const logicContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const activeKeys = displayOrder.filter(k => nodeActive[k]);
    if (activeKeys.length > 0) {
      setLatestActiveKey(activeKeys[activeKeys.length - 1]);
    }
  }, [nodeActive]);

  useEffect(() => {
    if (!latestActiveKey) return;
    const activeEl = logicRefs.current[latestActiveKey];
    const container = logicContainerRef.current;
    if (activeEl && container) {
      setTimeout(() => {
        const containerHeight = container.clientHeight;
        const itemOffsetTop = activeEl.offsetTop;
        const itemHeight = activeEl.offsetHeight;
        container.scrollTo({
          top: itemOffsetTop - containerHeight / 2 + itemHeight / 2,
          behavior: "smooth",
        });
      }, 0);
    }
  }, [latestActiveKey]);

  const activateNodeAndAncestors = useCallback((node: NodeKey) => {
    const newState = { ...nodeActive };
    function activate(n: NodeKey) {
      DAG[n].parents.forEach(activate);
      newState[n] = true;
    }
    activate(node);
    setNodeActive(newState);
  }, [nodeActive]);

  const deactivateNodeAndDescendants = useCallback((node: NodeKey) => {
    const newState = { ...nodeActive };
    function deactivate(n: NodeKey) {
      newState[n] = false;
      DAG[n].children.forEach(deactivate);
    }
    deactivate(node);
    setNodeActive(newState);
  }, [nodeActive]);

  // --- MAIN RETURN ---
  return (
    <div className="w-full min-h-[80vh] bg-gray-50 flex flex-col items-center justify-start">
    <div className="flex flex-col sm:flex-row w-fit max-w-full shadow">
      {/* ---- DIAGRAM COLUMN ---- */}
      <div className="flex-1 min-w-0 max-w-4xl px-1 xs:px-2 sm:px-4 md:px-8 py-3">

        {/* Input tokens */}
        <TokenMatrix tokens={[inputTokens]} />

        {/* Embedding Layer */}
        <LayerBox
          color={nodeDisplay.embedding.color}
          label={nodeDisplay.embedding.label}
          onPrev={() => deactivateNodeAndDescendants("embedding")}
          onNext={() => activateNodeAndAncestors("embedding")}
        >
          <div className="text-xs text-white w-full flex justify-center mt-1">
            <table className="border-separate border-spacing-x-4 xs:border-spacing-x-6">
              <tbody>
                {[...new Set(inputTokens)].sort().map((token, idx) => (
                  <tr key={idx}>
                    <td className="text-right text-white text-xs xs:text-sm">{token}</td>
                    <td className="text-center text-white">→</td>
                    <td className="text-left text-white text-xs xs:text-sm">{vocabulary[token]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </LayerBox>
        <TokenMatrix tokens={outputsToDisplay.embedding as string[][]} />

        {/* Decoder Layer (outer box) */}
        <div className="border-4 border-blue-400 bg-blue-50 rounded-3xl p-3 sm:p-8 w-full max-w-4xl mx-auto mb-6 xs:mb-8">
          <h2 className="text-xl xs:text-2xl font-bold text-center mb-5 xs:mb-7 text-blue-900 tracking-wide">
            Decoder Layer
          </h2>

          {/* Masked Self-Attention */}
          <div className="border border-amber-700 bg-amber-50 rounded-2xl p-2 sm:p-6 w-full max-w-4xl mb-4 sm:mb-6">
            <h3 className="text-lg xs:text-xl font-semibold text-center mb-2 xs:mb-4 text-amber-800">
              Masked Self-Attention
            </h3>
            {/* Responsive: stacks on small screens, side-by-side grid on >=sm */}
            <div
              className="
                flex flex-col
                sm:grid sm:grid-cols-3 sm:gap-x-2 sm:gap-y-4
                mt-2 w-full items-stretch
              "
            >
              {/* Query */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.query.color}
                  label={nodeDisplay.query.label}
                  onPrev={() => deactivateNodeAndDescendants("query")}
                  onNext={() => activateNodeAndAncestors("query")}
                />
                <TokenMatrix tokens={outputsToDisplay.query as string[][]} />
              </div>
              {/* Key */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.key.color}
                  label={nodeDisplay.key.label}
                  onPrev={() => deactivateNodeAndDescendants("key")}
                  onNext={() => activateNodeAndAncestors("key")}
                />
                <TokenMatrix tokens={outputsToDisplay.key as string[][]} />
              </div>
              {/* Value */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.value.color}
                  label={nodeDisplay.value.label}
                  onPrev={() => deactivateNodeAndDescendants("value")}
                  onNext={() => activateNodeAndAncestors("value")}
                />
                <TokenMatrix tokens={outputsToDisplay.value as string[][]} />
              </div>
              {/* Dot */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.dot.color}
                  label={nodeDisplay.dot.label}
                  onPrev={() => deactivateNodeAndDescendants("dot")}
                  onNext={() => activateNodeAndAncestors("dot")}
                />
                <TokenMatrix tokens={outputsToDisplay.dot as string[][]} />
              </div>
              {/* Argmax */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.argmax.color}
                  label={nodeDisplay.argmax.label}
                  onPrev={() => deactivateNodeAndDescendants("argmax")}
                  onNext={() => activateNodeAndAncestors("argmax")}
                />
                <TokenMatrix tokens={outputsToDisplay.argmax as string[][]} />
              </div>
              {/* Leftmost */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.leftmost.color}
                  label={nodeDisplay.leftmost.label}
                  onPrev={() => deactivateNodeAndDescendants("leftmost")}
                  onNext={() => activateNodeAndAncestors("leftmost")}
                />
                <TokenMatrix tokens={outputsToDisplay.leftmost as string[][]} />
              </div>
              {/* Weighted Sum */}
              <div className="flex flex-col items-center sm:col-span-3">
                <SubLayerBox
                  color={nodeDisplay.wsum.color}
                  label={nodeDisplay.wsum.label}
                  onPrev={() => deactivateNodeAndDescendants("wsum")}
                  onNext={() => activateNodeAndAncestors("wsum")}
                />
                <TokenMatrix tokens={outputsToDisplay.wsum as string[][]} />
              </div>
              {/* Projection */}
              <div className="flex flex-col items-center sm:col-span-3">
                <SubLayerBox
                  color={nodeDisplay.proj.color}
                  label={nodeDisplay.proj.label}
                  onPrev={() => deactivateNodeAndDescendants("proj")}
                  onNext={() => activateNodeAndAncestors("proj")}
                />
                <TokenMatrix tokens={outputsToDisplay.proj as string[][]} />
              </div>
            </div>
          </div>

          {/* Add & Norm 1 */}
          <LayerBox
            color={nodeDisplay.addnorm1.color}
            label={nodeDisplay.addnorm1.label}
            onPrev={() => deactivateNodeAndDescendants("addnorm1")}
            onNext={() => activateNodeAndAncestors("addnorm1")}
          />
          <TokenMatrix tokens={outputsToDisplay.addnorm1 as string[][]} />

          {/* Feed Forward sublayers - vertical */}
          <div className="border border-lime-700 bg-lime-50 rounded-2xl p-2 sm:p-6 w-full max-w-4xl mb-4 sm:mb-6 mt-2 sm:mt-4">
            <h3 className="text-lg xs:text-xl font-semibold text-center mb-2 xs:mb-4 text-lime-800">
              Feed Forward
            </h3>
            <div className="flex flex-col gap-2 sm:gap-4 w-full items-center">
              <div className="flex flex-col items-center w-full">
                <SubLayerBox
                  color={nodeDisplay.ff_linear1.color}
                  label={nodeDisplay.ff_linear1.label}
                  onPrev={() => deactivateNodeAndDescendants("ff_linear1")}
                  onNext={() => activateNodeAndAncestors("ff_linear1")}
                />
                <TokenMatrix tokens={outputsToDisplay.ff_linear1 as string[][]} />
              </div>
              <div className="flex flex-col items-center w-full">
                <SubLayerBox
                  color={nodeDisplay.ff_gelu.color}
                  label={nodeDisplay.ff_gelu.label}
                  onPrev={() => deactivateNodeAndDescendants("ff_gelu")}
                  onNext={() => activateNodeAndAncestors("ff_gelu")}
                />
                <TokenMatrix tokens={outputsToDisplay.ff_gelu as string[][]} />
              </div>
              <div className="flex flex-col items-center w-full">
                <SubLayerBox
                  color={nodeDisplay.ff_linear2.color}
                  label={nodeDisplay.ff_linear2.label}
                  onPrev={() => deactivateNodeAndDescendants("ff_linear2")}
                  onNext={() => activateNodeAndAncestors("ff_linear2")}
                />
                <TokenMatrix tokens={outputsToDisplay.ff_linear2 as string[][]} />
              </div>
            </div>
          </div>

          {/* Add & Norm 2 */}
          <LayerBox
            color={nodeDisplay.addnorm2.color}
            label={nodeDisplay.addnorm2.label}
            onPrev={() => deactivateNodeAndDescendants("addnorm2")}
            onNext={() => activateNodeAndAncestors("addnorm2")}
          />
          <TokenMatrix tokens={outputsToDisplay.addnorm2 as string[][]} />
        </div>

        {/* Linear */}
        <LayerBox
          color={nodeDisplay.linear.color}
          label={nodeDisplay.linear.label}
          onPrev={() => deactivateNodeAndDescendants("linear")}
          onNext={() => activateNodeAndAncestors("linear")}
        />
        <TokenMatrix tokens={outputsToDisplay.linear as string[][]} />
      </div>

      {/* --- LOGIC WALKTHROUGH SIDEBAR --- */}
      <aside
        className={`
          w-[15rem] xs:w-[18rem] sm:w-[22rem] md:w-[28rem] lg:w-[42rem]
          max-w-full h-[80vh] sticky top-0
          overflow-y-auto overflow-x-auto
          border-l pl-0 pr-0
          pt-3 xs:pt-4 sm:pt-6 md:pt-8
          flex-shrink-0
          text-[12px] xs:text-[13px] sm:text-[14px] md:text-[15px] lg:text-[16px]
        `}
        ref={logicContainerRef}
      >
        <div className="h-[40vh]" />
        <div className="space-y-4 xs:space-y-6 text-left leading-relaxed">
          {displayOrder.filter(k => nodeActive[k]).map((key) => (
            <div
              key={key}
              ref={(el) => { logicRefs.current[key] = el }}
              className={`transition-all px-2 xs:px-6 py-2 xs:py-4 rounded-lg ${
                key === latestActiveKey
                  ? "bg-blue-50 shadow-md font-semibold"
                  : "text-gray-700"
              }`}
            >
              {/* @ts-ignore */}
              {nodeLogics[key]}
            </div>
          ))}
        </div>
        <div className="h-[40vh]" />
      </aside>
    </div>
    </div>
  );
}
