import React from 'react';
import { MathJax } from 'better-react-mathjax';

// NodeKey definition (copy as-is to match your generic component)
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

// Node logics from your Example 2
const nodeLogics: Record<NodeKey, React.ReactNode> = {
  embedding: (
    <MathJax>
      {`$$
        \\begin{align}
        E_{-2.9}(x) &= \\pi_a(x) \\\\
        E_{0.05}(x) &= \\pi_b(x) \\\\
        E_{0}(x) &= \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  query: (
    <MathJax>
      {`$$
        \\begin{align}
        Q_{-1.3}(x) &= E_{-2.9}(x) \\\\
        &= \\pi_a(x) \\\\
        Q_{-0.2}(x) &= E_{0.05}(x) \\lor E_{0}(x) \\\\
        &= \\pi_b(x) \\lor \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  key: (
    <MathJax>
      {`$$
        \\begin{align}
        K_{-1.5}(x) &= E_{-2.9}(x) \\\\
        &= \\pi_a(x) \\\\
        K_{-0.4}(x) &= E_{0.05}(x) \\lor E_{0}(x) \\\\
        &= \\pi_b(x) \\lor \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  value: (
    <MathJax>
      {`$$
        \\begin{align}
        V_{-1.5}(x) &= E_{-2.9}(x) \\\\
        &= \\pi_a(x) \\\\
        V_{-0.6}(x) &= E_{0}(x) \\\\
        &= \\pi_c(x) \\\\
        V_{-0.4}(x) &= E_{0.05}(x) \\\\
        &= \\pi_b(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  dot: (
    <MathJax>
      {`$$
        \\begin{align}
        D_{0}(x,y) &= Q_{-0.2}(x) \\land K_{-0.4}(y) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        D_{0.3}(x,y) &= Q_{-0.2}(x) \\land K_{-1.5}(y) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(y) \\\\
        D_{0.5}(x,y) &= Q_{-1.3}(x) \\land K_{-0.4}(y) \\\\
        &= \\pi_a(x) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        D_{1.9}(x,y) &= Q_{-1.3}(x) \\land K_{-1.5}(y) \\\\
        &= \\pi_a(x) \\land \\pi_a(y)
        \\end{align}
      $$`}
    </MathJax>
  ),
  argmax: (
    <MathJax>
      {`$$
        \\begin{align}
        \\text{argmax}_{1.9}(x, y) &= D_{1.9}(x,y)\\\\
        &= \\pi_a(x) \\land \\pi_a(y) \\\\
        \\text{argmax}_{0.5}(x, y) &= D_{0.5}(x,y) \\land \\lnot \\exists z\\leq x:D_{1.9}(x,z) \\\\
        & = \\false \\\\
        \\text{argmax}_{0.3}(x, y) &= D_{0.3}(x,y) \\land \\lnot \\exists z\\leq x: (D_{1.9}(x,z) \\lor D_{0.5}(x,z)) \\\\
        &= (\\pi_{b}(x) \\lor \\pi_{c}(x)) \\land \\pi_{a}(y) \\\\
        \\text{argmax}_{0}(x, y) &= D_{0}(x,y) \\land \\lnot \\exists z\\leq x:(D_{1.9}(x,z)\\lor D_{0.5}(x,z)\\lor D_{0.3}(x,z)) \\\\
        &= (\\pi_b(y) \\lor \\pi_c(y)) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        \\text{argmax}(x, y) &= y\\leq x \\land (\\text{argmax}_{1.9}(x, y) \\lor \\text{argmax}_{0.5}(x, y) \\\\
        &\\quad\\quad\\quad\\quad\\quad \\lor \\text{argmax}_{0.3}(x, y) \\lor \\text{argmax}_{0}(x, y)) \\\\
        &= y\\leq x \\land (\\pi_a(y) \\lor \\lnot \\exists y \\leq x:\\pi_a(y))
        \\end{align}
      $$`}
    </MathJax>
  ),
  leftmost: (
    <MathJax>
      {`$$
        \\begin{align}
        \\text{leftmost}(x, y) &= \\text{argmax}(x, y) \\land \\lnot \\exists z<y: \\text{argmax}(x, z) \\\\
        &= y\\leq x \\land (\\pi_a(y)\\land (\\lnot \\exists z <y: \\pi_a(z)) \\\\
        &\\quad \\quad \\quad \\quad \\lor (\\lnot \\exists y\\leq x: \\pi_a(y))\\land \\lnot \\exists z<y)
        \\end{align}
      $$`}
    </MathJax>
  ),
  wsum: (
    <MathJax>
      {`$$
        \\begin{align}
        S_{-1.5}(x)&=\\exists y\\leq x: (\\text{leftmost}(x, y) \\land V_{-1.5}(y)) \\\\
        &= \\exists y\\leq x: (\\text{leftmost}(x, y) \\land \\pi_a(y)) \\\\
        &= \\exists y\\leq x: \\pi_a(y) \\\\
        S_{-0.6}(x)&=\\exists y\\leq x: (\\text{leftmost}(x, y) \\land V_{-0.6}(y)) \\\\
        &= \\exists y\\leq x: (\\text{leftmost}(x, y) \\land \\pi_c(y)) \\\\
        &= \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        S_{-0.4}(x)&=\\exists y\\leq x: (\\text{leftmost}(x, y) \\land V_{-0.4}(y)) \\\\
        &= \\exists y\\leq x: (\\text{leftmost}(x, y) \\land \\pi_b(y)) \\\\
        &= \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  proj: (
    <MathJax>
      {`$$
        \\begin{align}
        A_{-0.5}(x)&=S_{-1.5}(x) \\\\
        &= \\exists y\\leq x: \\pi_a(y) \\\\
        A_{-0.1}(x)&=S_{-0.6}(x) \\\\
        &= \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        A_{0.02}(x)&=S_{-0.4}(x) \\\\
        &= \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  addnorm1: (
    <MathJax>
      {`$$
        \\begin{align}
        N_{-3.5}(x) &= E_{-2.9}(x)\\land A_{-0.5}(x) \\\\
        &= \\pi_a(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        &= \\pi_a(x) \\\\
        N_{-0.5}(x) &= E_{0}(x) \\land A_{-0.5}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        N_{-0.4}(x) &= E_{0.05}(x) \\land A_{-0.5}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        N_{-0.1}(x) &= E_{0}(x) \\land A_{-0.1}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        N_{-0.06}(x) &= E_{0.05}(x) \\land A_{-0.1}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        N_{0.02}(x) &= E_{0}(x) \\land A_{0.02}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        N_{0.06}(x) &= E_{0.05}(x) \\land A_{0.02}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_linear1: (
    <MathJax>
      {`$$
        \\begin{align}
        L_{[-0.8, 5.6, -2.4, 1.4]}(x) &= N_{-3.5} \\\\
        &= \\pi_a(x) \\\\
        L_{[0, 0, -0.8, 0]}(x) &= N_{-0.1}(x) \\lor N_{-0.06}(x) \\lor N_{0.02} \\lor N_{0.06} \\\\
        &= \\lnot \\exists y\\leq x: \\pi_a(y) \\\\
        L_{[ 0, 1.4, -0.8, 0]}(x) &= N_{-0.5}(x) \\lor N_{-0.4}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_gelu: (
    <MathJax>
      {`$$
        \\begin{align}
        G_{[-0.2, 5.6, -0.03, 0]}(x) &= L_{[-0.8, 5.6, -2.4, 1.4]}(x)\\\\
        &= \\exists y\\leq x: ((\\pi_b(y) \\lor \\pi_c(y)) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        &= \\pi_a(x) \\\\
        G_{[0, 0, -0.2, 0]}(x) &= L_{[0, 0, -0.8, 0]}(x) \\lor L_{[ 0, 1.4, -0.8, 0]}(x) \\\\
        &= \\lnot \\exists y\\leq x: \\pi_a(y) \\lor (\\pi_b(x) \\lor \\pi_c(x)) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_linear2: (
    <MathJax>
      {`$$
        \\begin{align}
        F_{-3.1}(x) &= G_{[-0.2, 5.6, -0.03, 0]}(x) \\\\
        &= \\pi_a(x) \\\\
        F_{0}(x) &= G_{[0, 0, -0.2, 0]}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  addnorm2: (
    <MathJax>
      {`$$
        \\begin{align}
        R_{-6.6}(x) &= N_{-3.5}(x)\\land F_{-3.1}(x)\\\\
        &= \\pi_a(x) \\\\
        R_{0}(x) &= N_{-0.1}(x) \\land F_{0}(x) \\lor N_{-0.06}(x)\\land F_{0}(x) \\\\
        &\\quad \\lor N_{-0.4}(x) \\land F_{0}(x) \\lor N_{-0.5}(x) \\land F_{0}(x) \\\\
        & = \\big( \\lnot \\exists y\\leq x: \\pi_{a}(y)\\big) \\land  \\exists y \\leq x: \\Big( \\pi_{c}(y) \\land  \\lnot \\exists z2 < y \\Big) \\\\
        & \\quad \\lor (\\pi_{b}(x) \\lor \\pi_{c}(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        R_{0.03}(x) &= N_{0.02}(x) \\land F_{0}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        R_{0.06}(x) &= N_{0.06}(x) \\land F_{0}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y)
        \\end{align}
      $$`}
    </MathJax>
  ),
  linear: (
    <MathJax>
      {`$$
        \\begin{align}
        O_{-0.2}(x) &= R_{0.06}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        O_{-0.1}(x) &= R_{0.03}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        O_{-0.09}(x) &= R_{0}(x) \\\\\\
        & = \\big( \\lnot \\exists y\\leq x: \\pi_{a}(y)\\big) \\land  \\exists y \\leq x: \\Big( \\pi_{c}(y) \\land  \\lnot \\exists z2 < y \\Big) \\\\
        & \\quad \\lor (\\pi_{b}(x) \\lor \\pi_{c}(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        O_{10.8}(x) &= R_{-6.6}(x) \\\\
        &= \\pi_a(x) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
};

const inputTokens = ["b", "a", "c", "a", "c"];

const vocabulary: Record<string, number> = {
  b: 0.05,
  a: -2.9,
  c: 0,
};

const outputs: Record<NodeKey, string[][]> = {
  embedding: [["0.05", "-2.9", "0", "-2.9", "0"]],
  query: [["-0.2", "-1.3", "-0.2", "-1.3", "-0.2"]],
  key: [["-0.4", "-1.5", "-0.4", "-1.5", "-0.4"]],
  value: [["-0.4", "-1.5", "-0.6", "-1.5", "-0.6"]],
  dot: [
    ["0", "0.5", "0", "0.5", "0"],
    ["0.3", "1.9", "0.3", "1.9", "0.3"],
    ["0", "0.5", "0", "0.5", "0"],
    ["0.3", "1.9", "0.3", "1.9", "0.3"],
    ["0", "0.5", "0", "0.5", "0"],
  ],
  argmax: [
    ["1", "0", "0", "0", "0"],
    ["0", "1", "0", "0", "0"],
    ["0", "1", "0", "0", "0"],
    ["0", "1", "0", "1", "0"],
    ["0", "1", "0", "1", "0"],
  ],
  leftmost: [
    ["1", "0", "0", "0", "0"],
    ["0", "1", "0", "0", "0"],
    ["0", "1", "0", "0", "0"],
    ["0", "1", "0", "0", "0"],
    ["0", "1", "0", "0", "0"],
  ],
  wsum: [["-0.4", "-1.5", "-1.5", "-1.5", "-1.5"]],
  proj: [["0.02", "-0.5", "-0.5", "-0.5", "-0.5"]],
  addnorm1: [["0.06", "-3.5", "-0.5", "-3.5", "-0.5"]],
  ff_linear1: [
    ["0", "-0.8", "0", "-0.8", "0"],
    ["0", "5.6", "1.4", "5.6", "1.4"],
    ["-0.8", "-0.8", "0", "-0.8", "-0.8"],
    ["0", "1.4", "0", "1.4", "0"],
  ],
  ff_gelu: [
    ["0", "-0.2", "0", "-0.2", "0"],
    ["0", "5.6", "0", "5.6", "0"],
    ["-0.2", "-0.03", "-0.2", "-0.03", "-0.2"],
    ["0", "0", "0", "0", "0"],
  ],
  ff_linear2: [["0", "-3.1", "0", "-3.1", "0"]],
  addnorm2: [["0.06", "-6.6", "0", "-6.6", "0"]],
  linear: [["-0.2", "10.8", "-0.09", "47.4", "-0.09"]],
};

export const transformer2Data = {
  inputTokens,
  vocabulary,
  nodeLogics,
  outputs,
};
