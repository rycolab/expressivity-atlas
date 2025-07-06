import React from 'react';
import { MathJax } from 'better-react-mathjax';

// Must match the NodeKey used in the generic component
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

const nodeLogics: Record<NodeKey, React.ReactNode> = {
  embedding: (
    <MathJax>
      {`$$
        \\begin{align}
        E_{-11.5}(x) &= \\pi_a(x) \\\\
        E_{0.7}(x) &= \\pi_b(x) \\\\
        E_{0}(x) &= \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  query: (
    <MathJax>
      {`$$
        \\begin{align}
        Q_{-1.5}(x) &= E_{-11.5}(x) \\\\
        &= \\pi_a(x) \\\\
        Q_{-1.1}(x) &= E_{0.7}(x) \\lor E_{0}(x) \\\\
        &= \\pi_b(x) \\lor \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  key: (
    <MathJax>
      {`$$
        \\begin{align}
        K_{-0.3}(x) &= E_{-11.5}(x) \\\\
        &= \\pi_a(x) \\\\
        K_{-0.1}(x) &= E_{0.7}(x) \\lor E_{0}(x) \\\\
        &= \\pi_b(x) \\lor \\pi_c(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  value: (
    <MathJax>
      {`$$
        \\begin{align}
        V_{-9.7}(x) &= E_{-11.5}(x) \\\\
        &= \\pi_a(x) \\\\
        V_{2.3}(x) &= E_{0}(x) \\\\
        &= \\pi_c(x) \\\\
        V_{2.8}(x) &= E_{0.7}(x) \\\\
        &= \\pi_b(x)
        \\end{align}
      $$`}
    </MathJax>
  ),
  dot: (
    <MathJax>
      {`$$
        \\begin{align}
        D_{0.1}(x,y) &= Q_{-1.1}(x) \\land K_{-0.1}(y) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        D_{0.2}(x,y) &= Q_{-1.5}(x) \\land K_{-0.1}(y) \\\\
        &= \\pi_a(x) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        D_{0.3}(x,y) &= Q_{-1.1}(x) \\land K_{-0.3}(y) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(y) \\\\
        D_{0.5}(x,y) &= Q_{-1.5}(x) \\land K_{-0.3}(y) \\\\
        &= \\pi_a(x) \\land \\pi_a(y)
        \\end{align}
      $$`}
    </MathJax>
  ),
  argmax: (
    <MathJax>
      {`$$
        \\begin{align}
        \\text{argmax}_{0.5}(x, y) &= D_{0.5}(x,y)\\\\
        &= \\pi_a(x) \\land \\pi_a(y) \\\\
        \\text{argmax}_{0.3}(x, y) &= D_{0.3}(x,y) \\land \\lnot \\exists z\\leq x:D_{0.5}(x,z) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(y) \\land \\lnot \\exists z \\leq x:(\\pi_a(x) \\land \\pi_a(z)) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(y) \\\\
        \\text{argmax}_{0.2}(x, y) &= D_{0.2}(x,y) \\land \\lnot \\exists z\\leq x:(D_{0.5}(x,z)\\lor D_{0.3}(x,z)) \\\\
        &= \\pi_a(x) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        &\\quad \\land \\lnot \\exists z \\leq x:(\\pi_a(x) \\land \\pi_a(z) \\lor (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(z) ) \\\\
        &=\\false \\\\
        \\text{argmax}_{0.1}(x, y) &= D_{0.1}(x,y) \\land \\lnot \\exists z\\leq x:(D_{0.5}(x,z)\\lor D_{0.3}(x,z)\\lor D_{0.2}(x,z)) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\\\
        &\\quad \\land \\lnot \\exists z \\leq x:(\\pi_a(x) \\land \\pi_a(z) \\lor (\\pi_b(x) \\lor \\pi_c(x)) \\land \\pi_a(z)  \\\\
        &\\quad\\quad\\quad\\quad\\quad\\quad \\lor \\pi_a(x) \\land (\\pi_b(z) \\lor \\pi_c(z)) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land (\\pi_b(y) \\lor \\pi_c(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z)  \\\\
        &= (\\pi_b(y) \\lor \\pi_c(y)) \\land \\lnot \\exists z \\leq x:\\pi_a(z) \\\\
        &= (\\pi_b(y) \\lor \\pi_c(y)) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        \\text{argmax}(x, y) &= y\\leq x \\land (\\text{argmax}_{0.5}(x, y) \\lor \\text{argmax}_{0.3}(x, y) \\\\
        &\\quad\\quad\\quad\\quad\\quad \\lor \\text{argmax}_{0.2}(x, y) \\lor \\text{argmax}_{0.1}(x, y)) \\\\
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
        S_{-9.7}(x)&=\\exists y\\leq x: (\\text{leftmost}(x, y) \\land V_{-9.7}(y)) \\\\
        &= \\exists y\\leq x: (\\text{leftmost}(x, y) \\land \\pi_a(y)) \\\\
        &= \\exists y\\leq x: \\pi_a(y) \\\\
        S_{2.3}(x)&=\\exists y\\leq x: (\\text{leftmost}(x, y) \\land V_{2.3}(y)) \\\\
        &= \\exists y\\leq x: (\\text{leftmost}(x, y) \\land \\pi_c(y)) \\\\
        &= \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        S_{2.8}(x)&=\\exists y\\leq x: (\\text{leftmost}(x, y) \\land V_{2.8}(y)) \\\\
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
        A_{-20.8}(x)&=S_{-9.7}(x) \\\\
        &= \\exists y\\leq x: \\pi_a(y) \\\\
        A_{6.1}(x)&=S_{2.3}(x) \\\\
        &= \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        A_{8.1}(x)&=S_{2.8}(x) \\\\
        &= \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  addnorm1: (
    <MathJax>
      {`$$
        \\begin{align}
        N_{-32.3}(x) &= E_{-11.5}(x)\\land A_{-20.8}(x) \\\\
        &= \\pi_a(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        &= \\pi_a(x) \\\\
        N_{-20.8}(x) &= E_{0}(x) \\land A_{-20.8}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        N_{-20.1}(x) &= E_{0.7}(x) \\land A_{-20.8}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        N_{6.1}(x) &= E_{0}(x) \\land A_{6.1}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        N_{6.8}(x) &= E_{0.7}(x) \\land A_{6.1}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        N_{8.1}(x) &= E_{0}(x) \\land A_{8.1}(x) \\\\
        &= \\pi_c(x) \\land \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        N_{8.8}(x) &= E_{0.7}(x) \\land A_{8.1}(x) \\\\
        &= \\pi_b(x) \\land \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_linear1: (
    <MathJax>
      {`$$
        \\begin{align}
        L_{[-16.0, 156.5, 0, 0]}(x) &= N_{-20.8}(x) \\lor N_{-20.1}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        L_{[-16.0, 234.8, 78.3, 0]}(x) &= N_{-32.3}(x) \\\\
        &= \\pi_a(x) \\\\
        L_{[ 0, -63.9, -16.0, -16.0]}(x) &= N_{8.1}(x) \\lor N_{8.8}(x) \\\\
        &= \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        L_{[ 0, -48.0, -16.0, 0]}(x) &= N_{6.1}(x) \\lor N_{6.8}(x) \\\\
        &= \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_gelu: (
    <MathJax>
      {`$$
        \\begin{align}
        G_{[0, 0, 0, 0]}(x) &= L_{[ 0, -63.9, -16.0, -16.0]}(x) \\lor L_{[ 0, -48.0, -16.0, 0]}(x) \\\\
        &= \\exists y\\leq x: ((\\pi_b(y) \\lor \\pi_c(y)) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        &= \\lnot \\exists z\\leq x: \\pi_a(z) \\\\
        G_{[0, 167.7, 0, 0]}(x) &= L_{[-16.0, 156.5, 0, 0]}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        G_{[ 0, 234.8, 67.1, 0]}(x) &= L_{[-16.0, 234.8, 78.3, 0]}(x) \\\\
        &= \\pi_a(x) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  ff_linear2: (
    <MathJax>
      {`$$
        \\begin{align}
        F_{-67.6}(x) &= G_{[ 0, 234.8, 67.1, 0]}(x) \\\\
        &= \\pi_a(x) \\\\
        F_{-45.1}(x) &= G_{[0, 167.7, 0, 0]}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        F_{0.7}(x) &= G_{[0, 0, 0, 0]}(x) \\\\
        &= \\lnot \\exists z\\leq x: \\pi_a(z) \\\\
        \\end{align}
      $$`}
    </MathJax>
  ),
  addnorm2: (
    <MathJax>
      {`$$
        \\begin{align}
        R_{-99.9}(x) &= N_{-32.3}(x)\\land F_{-67.6}(x)\\\\
        &= \\pi_a(x) \\\\
        R_{-66.6}(x) &= N_{-20.8}(x) \\land F_{-45.1}(x) \\lor N_{-20.1}(x)\\land F_{-45.1}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        R_{7.1}(x) &= N_{6.1}(x) \\land F_{0.7}(x) \\lor N_{6.8}(x)\\land F_{0.7}(x) \\\\
        &= \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        R_{9.4}(x) &= N_{8.1}(x) \\land F_{0.7}(x) \\lor N_{8.8}(x)\\land F_{0.7}(x) \\\\
        &= \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y)
        \\end{align}
      $$`}
    </MathJax>
  ),
  linear: (
    <MathJax>
      {`$$
        \\begin{align}
        O_{-5.3}(x) &= R_{9.4}(x) \\\\
        &= \\exists y\\leq x: (\\pi_b(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        O_{-4.0}(x) &= R_{7.1}(x) \\\\
        &= \\exists y\\leq x: (\\pi_c(y) \\land \\lnot \\exists z < y) \\land \\lnot \\exists y \\leq x:\\pi_a(y) \\\\
        O_{-5.3}(x) \\lor O_{-4.0}(x) &= \\lnot \\exists z \\leq x:\\pi_a(z) \\\\\\
        O_{31.6}(x) &= R_{-66.6}(x) \\\\
        &= (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        O_{47.4}(x) &= R_{-99.9}(x) \\\\
        O_{31.6}(x) \\lor O_{47.4}(x) &= \\pi_a(x) \\lor (\\pi_b(x) \\lor \\pi_c(x)) \\land \\exists y\\leq x: \\pi_a(y) \\\\
        &= \\pi_a(x) \\lor \\exists y\\leq x: \\pi_a(y) \\\\
        &= \\exists y\\leq x: \\pi_a(y)
        \\end{align}
      $$`}
    </MathJax>
  ),
};

const inputTokens = ["b", "a", "c", "a", "c"];

const vocabulary: Record<string, number> = {
  b: 0.7,
  a: -11.5,
  c: 0,
};

const outputs: Record<NodeKey, string[][]> = {
  embedding: [["0.7", "-11.5", "0", "-11.5", "0"]],
  query: [["-1.1", "-1.5", "-1.1", "-1.5", "-1.1"]],
  key: [["-0.1", "-0.3", "-0.1", "-0.3", "-0.1"]],
  value: [["2.8", "-9.7", "2.3", "-9.7", "2.3"]],
  dot: [
    ["0.1", "0.3", "0.1", "0.3", "0.1"],
    ["0.2", "0.5", "0.2", "0.5", "0.2"],
    ["0.1", "0.3", "0.1", "0.3", "0.1"],
    ["0.2", "0.5", "0.2", "0.5", "0.2"],
    ["0.1", "0.3", "0.1", "0.3", "0.1"],
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
  wsum: [["2.8", "-9.7", "-9.7", "-9.7", "-9.7"]],
  proj: [["8.1", "-20.8", "-20.8", "-20.8", "-20.8"]],
  addnorm1: [["8.1", "-32.3", "-20.8", "-32.3", "-20.8"]],
  ff_linear1: [
    ["0", "-16.0", "-16.0", "-16.0", "-16.0"],
    ["-63.9", "234.8", "156.5", "234.8", "156.5"],
    ["-16.0", "78.3", "0", "78.3", "0"],
    ["-16.0", "0", "0", "0", "0"],
  ],
  ff_gelu: [
    ["0", "0", "0", "0", "0"],
    ["0", "234.8", "167.7", "234.8", "167.7"],
    ["0", "67.1", "0", "67.1", "0"],
    ["0", "0", "0", "0", "0"],
  ],
  ff_linear2: [["0.7", "-67.6", "-45.1", "-67.6", "-45.1"]],
  addnorm2: [["9.4", "-99.9", "-66.6", "-99.9", "-66.6"]],
  linear: [["-5.3", "47.4", "31.6", "47.4", "31.6"]],
};

export const transformer1Data = {
  inputTokens,
  vocabulary,
  nodeLogics,
  outputs,
};
