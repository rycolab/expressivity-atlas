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
  | "softmax"
  | "wsum"
  | "addnorm1"
  | "ff"
  | "addnorm2"

  | "query2"
  | "key2"
  | "value2"
  | "dot2"
  | "softmax2"
  | "wsum2"
  | "addnorm3"
  | "ff2"
  | "addnorm4"

  | "linear"

const DAG: Record<NodeKey, { parents: NodeKey[]; children: NodeKey[] }> = {
  embedding: { parents: [], children: ["query", "key", "value"] },
  query: { parents: ["embedding"], children: ["dot"] },
  key: { parents: ["embedding"], children: ["dot"] },
  value: { parents: ["embedding"], children: ["wsum"] },
  dot: { parents: ["query", "key"], children: ["softmax"] },
  softmax: { parents: ["dot"], children: ["wsum"] },
  wsum: { parents: ["softmax", "value"], children: ["addnorm1"] },
  addnorm1: { parents: ["wsum"], children: ["ff"] },
  ff: { parents: ["addnorm1"], children: ["addnorm2"]},
  addnorm2: { parents: ["ff"], children: ["query2", "key2", "value2"] },
  query2: { parents: ["addnorm2"], children: ["dot2"] },
  key2: { parents: ["addnorm2"], children: ["dot2"] },
  value2: { parents: ["addnorm2"], children: ["wsum2"] },
  dot2: { parents: ["query2", "key2"], children: ["softmax2"] },
  softmax2: { parents: ["dot2"], children: ["wsum2"] },
  wsum2: { parents: ["softmax2", "value2"], children: ["addnorm3"] },
  addnorm3: { parents: ["wsum2"], children: ["ff2"] },
  ff2: { parents: ["addnorm3"], children: ["addnorm4"]},
  addnorm4: { parents: ["ff2"], children: ["linear"] },
  linear: { parents: ["addnorm4"], children: [] },
};

const nodeDisplay: Record<NodeKey, { label: string; color: string }> = {
  embedding: { label: "Embedding", color: "bg-blue-700" },
  query: { label: "Query projection", color: "bg-amber-600" },
  key: { label: "Key projection", color: "bg-amber-600" },
  value: { label: "Value projection", color: "bg-amber-600" },
  dot: { label: "Scaled Dot-Product", color: "bg-amber-600" },
  softmax: { label: "Softmax", color: "bg-amber-600" },
  wsum: { label: "Weighted sum", color: "bg-amber-600" },
  addnorm1: { label: "Add", color: "bg-slate-600" },
  ff: { label: "Feed Forward", color: "bg-lime-600" },
  addnorm2: { label: "Add", color: "bg-slate-600" },
  query2: { label: "Query projection", color: "bg-amber-600" },
  key2: { label: "Key projection", color: "bg-amber-600" },
  value2: { label: "Value projection", color: "bg-amber-600" },
  dot2: { label: "Scaled Dot-Product", color: "bg-amber-600" },
  softmax2: { label: "Softmax", color: "bg-amber-600" },
  wsum2: { label: "Weighted sum", color: "bg-amber-600" },
  addnorm3: { label: "Add", color: "bg-slate-600" },
  ff2: { label: "Feed Forward", color: "bg-lime-600" },
  addnorm4: { label: "Add", color: "bg-slate-600" },
  linear: { label: "Linear", color: "bg-rose-600" },
};


// displayOrder remains unchanged
const displayOrder: NodeKey[] = [
  "embedding",
  "query", "key", "value",
  "dot", "softmax", "wsum",
  "addnorm1",
  "ff",
  "addnorm2",
  "query2", "key2", "value2",
  "dot2", "softmax2", "wsum2",
  "addnorm3",
  "ff2",
  "addnorm4",
  "linear",
];

type TokenBoxProps = {
  token: string | { value: string; highlight?: boolean };
};

const TokenBox = ({ token }: TokenBoxProps) => {
  const isObj = typeof token === "object";
  const value = isObj ? token.value : token;
  const highlight = isObj && token.highlight;

  return (
    <div
      className={`
        border rounded-xl shadow-sm text-center
        flex items-center justify-center
        transition-all
        w-6 h-6 text-[7px]
        xs:w-7 xs:h-7 xs:text-[8px]
        sm:w-8 sm:h-8 sm:text-[9px]
        md:w-9 md:h-9 md:text-[10px]
        lg:w-10 lg:h-10 lg:text-[11px]
        max-w-[2rem] max-h-[2rem]
        break-all
        ${highlight ? "bg-yellow-200 border-yellow-400 font-bold" : "bg-white"}
      `}
      style={{
        minWidth: "1.5rem",
        minHeight: "1.5rem",
      }}
    >
      {value}
    </div>
  );
};

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


const vocabulary: Record<string, number[]> = {
  a: [1,0,0,0,0,0,0],
  b: [0,1,0,0,0,0,0],
  c: [0,0,1,0,0,0,0],
};
const makeMatrix = (rows: number, cols: number, prefix: string) =>
  Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) => prefix ? `${prefix}${i + 1}${j + 1}` : "")
  );

export default function TransformerDiagram2() {
  const inputTokens = ["b", "a", "c", "a", "c"];

  const [nodeActive, setNodeActive] = useState<Record<NodeKey, boolean>>({
    embedding: false,
    query: false,
    key: false,
    value: false,
    dot: false,
    softmax: false,
    wsum: false,
    addnorm1: false,
    ff: false,
    addnorm2: false,
    query2: false,
    key2: false,
    value2: false,
    dot2: false,
    softmax2: false,
    wsum2: false,
    addnorm3: false,
    ff2: false,
    addnorm4: false,
    linear: false
  });

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
        const offset = containerHeight * 0.4;
        container.scrollTo({
          top: itemOffsetTop - containerHeight / 2 + itemHeight / 2 + offset,
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

  const [alternateMode, setAlternateMode] = useState(false);
  const [showFullDiagram, setShowFullDiagram] = useState(false);

  // (outputs definition unchanged)
  const outputs = useMemo(() => ({
    embedding: nodeActive.embedding
      ? [
        ["0", "1", "0", "1", "0"],
        ["1", "0", "0", "0", "0"],
        ["0", "0", "1", "0", "1"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    query: nodeActive.query
      ? [
        ["-800", "-800", "-800", "-800", "-800"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    key: nodeActive.key
      ? [
        ["-1", "0", "-1", "0", "-1"],
        ["0", "-1", "-1", "-1", "-1"],
        ["-1", "-1", "0", "-1", "0"],
        ["-1", "-1", "-1", "-1", "-1"],
        ["-1", "-1", "-1", "-1", "-1"],
        ["-1", "-1", "-1", "-1", "-1"],
        ["-1", "-1", "-1", "-1", "-1"]
      ]
      : makeMatrix(7, 5, ""),
    value: nodeActive.value
      ? [
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "1", "0", "1", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    dot: nodeActive.dot
      ? [
        ["-800", "0", "-800", "0", "-800"],
        ["-800", "0", "-800", "0", "-800"],
        ["-800", "0", "-800", "0", "-800"],
        ["-800", "0", "-800", "0", "-800"],
        ["-800", "0", "-800", "0", "-800"]
      ]
      : makeMatrix(5, 5, ""),
    softmax: nodeActive.softmax
      ? alternateMode
        ? [
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "1", "0", "0", "0"],
            ["0", "1", "0", "0", "0"],
            ["0", {value: "0", highlight: true}, "0", {value: "0", highlight: true}, "0"]
          ]
        : [
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "1", "0", "0", "0"],
            ["0", "1", "0", "0", "0"],
            ["0", "0.5", "0", "0.5", "0"]
          ]
      : makeMatrix(5, 5, ""),

    wsum: nodeActive.wsum
      ? alternateMode
        ? [
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "1", "1", {value: "0", highlight: true}],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"]
          ]
        : [
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "1", "1", "1"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"]
          ]
      : makeMatrix(7, 5, ""),

    addnorm1: nodeActive.addnorm1
      ? alternateMode
        ? [
          ["0", "1", "0", "1", "0"],
          ["1", "0", "0", "0", "0"],
          ["0", "0", "1", "0", "1"],
          ["0", "0", "1", "1", { value: "0", highlight: true} ],
          ["0", "0", "0", "0", "0"],
          ["0", "0", "0", "0", "0"],
          ["0", "0", "0", "0", "0"]
        ]
        : [
            ["0", "1", "0", "1", "0"],
            ["1", "0", "0", "0", "0"],
            ["0", "0", "1", "0", "1"],
            ["0", "0", "1", "1", "1"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"],
            ["0", "0", "0", "0", "0"]
          ]
      : makeMatrix(7, 5, ""),
    ff: nodeActive.ff
      ? [
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "1", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    addnorm2: nodeActive.addnorm2
      ? [
        ["0", "1", "0", "1", "0"],
        ["1", "0", "0", "0", "0"],
        ["0", "0", "1", "0", "1"],
        ["0", "0", "1", "1", "0"],
        ["0", "0", "0", "1", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    query2: nodeActive.query2
      ? [
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["-800", "-800", "-800", "-800", "-800"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    key2: nodeActive.key2
      ? [
        ["-1", "0", "-1", "0", "-1"],
        ["0", "-1", "-1", "-1", "-1"],
        ["-1", "-1", "0", "-1", "0"],
        ["-1", "-1", "0", "0", "-1"],
        ["-1", "-1", "-1", "0", "-1"],
        ["-1", "-1", "-1", "-1", "-1"],
        ["-1", "-1", "-1", "-1", "-1"]
      ]
      : makeMatrix(7, 5, ""),
    value2: nodeActive.value2
      ? [
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "1", "0"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    dot2: nodeActive.dot2
      ? [
        ["-800", "-800", "-800", "0", "-800"],
        ["-800", "-800", "-800", "0", "-800"],
        ["-800", "-800", "-800", "0", "-800"],
        ["-800", "-800", "-800", "0", "-800"],
        ["-800", "-800", "-800", "0", "-800"]
      ]
      : makeMatrix(5, 5, ""),
    softmax2: nodeActive.softmax2
      ? [
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "1", "0"]
      ]
      : makeMatrix(5, 5, ""),
    wsum2: nodeActive.wsum2
      ? [
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "1"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    addnorm3: nodeActive.addnorm3
      ? [
        ["0", "1", "0", "1", "0"],
        ["1", "0", "0", "0", "0"],
        ["0", "0", "1", "0", "1"],
        ["0", "0", "1", "1", "0"],
        ["0", "0", "0", "1", "0"],
        ["0", "0", "0", "0", "1"],
        ["0", "0", "0", "0", "0"]
      ]
      : makeMatrix(7, 5, ""),
    ff2: nodeActive.ff2
      ? [
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "0", "0", "0"],
        ["0", "0", "1", "1", "1"]
      ]
      : makeMatrix(7, 5, ""),
    addnorm4: nodeActive.addnorm4
      ? [
        ["0", "1", "0", "1", "0"],
        ["1", "0", "0", "0", "0"],
        ["0", "0", "1", "0", "1"],
        ["0", "0", "1", "1", "0"],
        ["0", "0", "0", "1", "0"],
        ["0", "0", "0", "0", "1"],
        ["0", "0", "1", "1", "1"]
      ]
      : makeMatrix(7, 5, ""),
    linear: nodeActive.linear
      ? [["0", "0", "1", "1", "1"]]
      : makeMatrix(1, 5, ""),
  }), [nodeActive, inputTokens]);

  const nodeLogics: Record<NodeKey, React.ReactNode> = {
    embedding: (
      "Assign each unique token a distinct embedding by activating a corresponding dimension in a one-hot manner."
    ),
    query: (
      "Assign a large positive value (e.g., 800) at the dimension of interest, and 0 elsewhere. This ensures that when combined in attention, only the relevant key stands out."
    ),
    key: (
      <MathJax>
        {`$$
          \\mK=\\mE-1
        $$`}
      </MathJax>
    ),
    value: (
      "Copy the target dimension (the 1st) into an unused one (the 4th) to carry its value forward. All other dimensions are zero."
    ),
    dot: (
      <MathJax>
        {`Dot product results in 0 at positions where $\\pi_a$ is satisfied, and -800 elsewhere.`}
      </MathJax>
    ),
    softmax: (
      <MathJax>
        {alternateMode
          ? "The last row becomes all zeros because two earlier positions satisfy $\\pi_a$, exceeding the limit of one nonzero entry and diluting the attention."
          : "Acts effectively the same as average hard attention."
        }
      </MathJax>
    ),
    wsum: (
      <MathJax>
        {alternateMode
          ? "This results in a flawed simulation of $\\past\\pi_a$, which is incorrect at the final position."
          : "The new dimension now simulates $\\past\\pi_a$."
        }
      </MathJax>
    ),
    addnorm1: (
      <div className="space-y-2">
        {alternateMode ? (
          <>
            <MathJax>
              {"The current simulation is flawed and needs correction."}
            </MathJax>
            {!showFullDiagram && (
              <button
                onClick={() => setShowFullDiagram(true)}
                className="text-xs px-3 py-1 bg-white/70 text-gray-900 border border-gray-300 rounded hover:bg-white shadow-sm hover:shadow transition cursor-pointer"
              >
                Add an additional layer to fix the simulation →
              </button>
            )}
          </>
        ) : (
          <>
            <MathJax>
              {"Job done? Not quite. When too many positions satisfy $\\past \\pi_a$, attention weights vanish."}
            </MathJax>
              <button
                onClick={() => {
                  setAlternateMode(true);
                  deactivateNodeAndDescendants("wsum");
                }}
                className="text-xs px-3 py-1 bg-white/70 text-gray-900 border border-gray-300 rounded hover:bg-white shadow-sm hover:shadow transition cursor-pointer"
              >
                Assume only one nonzero entry in softmax →
              </button>
          </>
        )}
      </div>
    ),
    ff: (
      <MathJax>
        {`As a preparatory step, we we compute the logical intersection of $\\pi_a$ from dimension 1 and the flawed $\\past\\pi_a$ from dimension 4, and store the result in a new dimension:
        $$
        \\mF_{5,:} = \\ReLU(\\mN_{1,:} + \\mN_{4,:})
        $$
        Only one position satisfies this condition at most.
        `}
      </MathJax>
    ),
    addnorm2: (
      "Adds the feedforward output to the residual from the previous sublayer."
    ),
    query2: (
      "Similar to the first attention, but now targeting the 5th dimension."
    ),
    key2: (
      <MathJax>
        {`$$
          \\mK=\\mR-1
        $$`}
      </MathJax>
    ),
    value2: (
      "Copy the target dimension (the 5th) into an unused one (the 6th) to carry its value forward. All other dimensions are zero."
    ),
    dot2: (
      <MathJax>
        {`The dot product yields 0 at positions where $\\mR_{5,:} = 1$, and -800 elsewhere. As noted, at most one such position exists.`}
      </MathJax>
    ),
    softmax2: (
      "Acts effectively the same as average hard attention. Attention weights never vanish here."
    ),
    wsum2: (
      <MathJax>
        {`
          The new dimension now simulates $\\past(\\pi_a\\land \\past\\pi_a)$. It's different from $\\past\\pi_a$ at positions where there is only one a before it.
        `}
      </MathJax>
    ),
    addnorm3: (
      <MathJax>
        {`
          Residual connection.
        `}
      </MathJax>
    ),
    ff2: (
      <MathJax>
        {`Compute the logical union of the correct $\\past(\\pi_a \\land \\past\\pi_a)$ (dim 6) and the flawed $\\past\\pi_a$ (dim 4), and store the result in the 7th dimension:
        $$
        \\mF_{7,:} = 1 - \\ReLU(1 - \\mN_{4,:} - \\mN_{6,:})
        $$
        This results in a correct simulation of $\\past\\pi_a$.
        `}
      </MathJax>
    ),
    addnorm4: (
      "Adds the feedforward output to the residual from the previous sublayer."
    ),
    linear: (
      <MathJax>
        {`The linear layer just copies the last dimension.`}
      </MathJax>
    ),
  };
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
                    <td className="text-left text-white text-xs xs:text-sm">
                      [{vocabulary[token].join(', ')}]
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </LayerBox>
        <TokenMatrix tokens={outputs.embedding as string[][]} />

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
                <TokenMatrix tokens={outputs.query as string[][]} />
              </div>
              {/* Key */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.key.color}
                  label={nodeDisplay.key.label}
                  onPrev={() => deactivateNodeAndDescendants("key")}
                  onNext={() => activateNodeAndAncestors("key")}
                />
                <TokenMatrix tokens={outputs.key as string[][]} />
              </div>
              {/* Value */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.value.color}
                  label={nodeDisplay.value.label}
                  onPrev={() => deactivateNodeAndDescendants("value")}
                  onNext={() => activateNodeAndAncestors("value")}
                />
                <TokenMatrix tokens={outputs.value as string[][]} />
              </div>
              {/* Dot */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.dot.color}
                  label={nodeDisplay.dot.label}
                  onPrev={() => deactivateNodeAndDescendants("dot")}
                  onNext={() => activateNodeAndAncestors("dot")}
                />
                <TokenMatrix tokens={outputs.dot as string[][]} />
              </div>
              {/* Softmax */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.softmax.color}
                  label={nodeDisplay.softmax.label}
                  onPrev={() => deactivateNodeAndDescendants("softmax")}
                  onNext={() => activateNodeAndAncestors("softmax")}
                />
                <TokenMatrix tokens={outputs.softmax as string[][]} />
              </div>
              {/* Weighted Sum */}
              <div className="flex flex-col items-center sm:col-span-3">
                <SubLayerBox
                  color={nodeDisplay.wsum.color}
                  label={nodeDisplay.wsum.label}
                  onPrev={() => deactivateNodeAndDescendants("wsum")}
                  onNext={() => activateNodeAndAncestors("wsum")}
                />
                <TokenMatrix tokens={outputs.wsum as string[][]} />
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
          <TokenMatrix tokens={outputs.addnorm1 as string[][]} />

          {showFullDiagram && (
            <>
          {/* Feed Forward sublayers - vertical */}
          <LayerBox
            color={nodeDisplay.ff.color}
            label={nodeDisplay.ff.label}
            onPrev={() => deactivateNodeAndDescendants("ff")}
            onNext={() => activateNodeAndAncestors("ff")}
          />
          <TokenMatrix tokens={outputs.ff as string[][]} />

          {/* Add & Norm 2 */}
          <LayerBox
            color={nodeDisplay.addnorm2.color}
            label={nodeDisplay.addnorm2.label}
            onPrev={() => deactivateNodeAndDescendants("addnorm2")}
            onNext={() => activateNodeAndAncestors("addnorm2")}
          />
          <TokenMatrix tokens={outputs.addnorm2 as string[][]} />
          </>
          )}
        </div>

        {showFullDiagram && (
          <>
        <div className="border-4 border-blue-400 bg-blue-50 rounded-3xl p-3 sm:p-8 w-full max-w-4xl mx-auto mb-6 xs:mb-8">
                <h2 className="text-xl xs:text-2xl font-bold text-center mb-5 xs:mb-7 text-blue-900 tracking-wide">
                  Decoder Layer 2
                </h2>
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
                  color={nodeDisplay.query2.color}
                  label={nodeDisplay.query2.label}
                  onPrev={() => deactivateNodeAndDescendants("query2")}
                  onNext={() => activateNodeAndAncestors("query2")}
                />
                <TokenMatrix tokens={outputs.query2 as string[][]} />
              </div>
              {/* Key */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.key2.color}
                  label={nodeDisplay.key2.label}
                  onPrev={() => deactivateNodeAndDescendants("key2")}
                  onNext={() => activateNodeAndAncestors("key2")}
                />
                <TokenMatrix tokens={outputs.key2 as string[][]} />
              </div>
              {/* Value */}
              <div className="flex flex-col items-center mb-2 sm:mb-0">
                <SubLayerBox
                  color={nodeDisplay.value2.color}
                  label={nodeDisplay.value2.label}
                  onPrev={() => deactivateNodeAndDescendants("value2")}
                  onNext={() => activateNodeAndAncestors("value2")}
                />
                <TokenMatrix tokens={outputs.value2 as string[][]} />
              </div>
              {/* Dot */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.dot2.color}
                  label={nodeDisplay.dot2.label}
                  onPrev={() => deactivateNodeAndDescendants("dot2")}
                  onNext={() => activateNodeAndAncestors("dot2")}
                />
                <TokenMatrix tokens={outputs.dot2 as string[][]} />
              </div>
              {/* Softmax */}
              <div className="flex flex-col items-center sm:col-span-2">
                <SubLayerBox
                  color={nodeDisplay.softmax2.color}
                  label={nodeDisplay.softmax2.label}
                  onPrev={() => deactivateNodeAndDescendants("softmax2")}
                  onNext={() => activateNodeAndAncestors("softmax2")}
                />
                <TokenMatrix tokens={outputs.softmax2 as string[][]} />
              </div>
              {/* Weighted Sum */}
              <div className="flex flex-col items-center sm:col-span-3">
                <SubLayerBox
                  color={nodeDisplay.wsum2.color}
                  label={nodeDisplay.wsum2.label}
                  onPrev={() => deactivateNodeAndDescendants("wsum2")}
                  onNext={() => activateNodeAndAncestors("wsum2")}
                />
                <TokenMatrix tokens={outputs.wsum2 as string[][]} />
              </div>
            </div>
          </div>

          {/* Add & Norm 1 */}
          <LayerBox
            color={nodeDisplay.addnorm3.color}
            label={nodeDisplay.addnorm3.label}
            onPrev={() => deactivateNodeAndDescendants("addnorm3")}
            onNext={() => activateNodeAndAncestors("addnorm3")}
          />
          <TokenMatrix tokens={outputs.addnorm3 as string[][]} />

          {/* Feed Forward sublayers - vertical */}
          <LayerBox
            color={nodeDisplay.ff2.color}
            label={nodeDisplay.ff2.label}
            onPrev={() => deactivateNodeAndDescendants("ff2")}
            onNext={() => activateNodeAndAncestors("ff2")}
          />
          <TokenMatrix tokens={outputs.ff2 as string[][]} />

          {/* Add & Norm 2 */}
          <LayerBox
            color={nodeDisplay.addnorm4.color}
            label={nodeDisplay.addnorm4.label}
            onPrev={() => deactivateNodeAndDescendants("addnorm4")}
            onNext={() => activateNodeAndAncestors("addnorm4")}
          />
          <TokenMatrix tokens={outputs.addnorm4 as string[][]} />
        </div>

        {/* Linear */}
        <LayerBox
          color={nodeDisplay.linear.color}
          label={nodeDisplay.linear.label}
          onPrev={() => deactivateNodeAndDescendants("linear")}
          onNext={() => activateNodeAndAncestors("linear")}
        />
        <TokenMatrix tokens={outputs.linear as string[][]} />
        </>
        )}
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




{/* <h2 className="text-2xl font-bold text-center mb-6 tracking-tight">
  Equivalent Logical Expressions
</h2> */}
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

// Place your nodeLogics, vocabulary, and makeMatrix functions as in your original file!
