# Expressivity Atlas

## 🚀 Getting Started

### 1. Prerequisites

Make sure you have [Node.js](https://nodejs.org) installed.

Then install [pnpm](https://pnpm.io):

```bash
npm install -g pnpm
```

### 2. Install dependencies

Install project dependencies:

```bash
pnpm i
```

### 3. Run the development server

```bash
pnpm dev
```

Visit `http://localhost:3000` in your browser.

## 🧱 Project Structure

- Pages are located in subfolders of `/app`, such as `/automata`, `/circuits`, `/languages`, `/networks`.
- Shared layouts and components are under `app/ui/`.
- Graph data is defined in `app/data/graphData.ts`.
- MathJax macros are defined in `app/mathjax-config.ts`.

## ➕ Adding a New Page

To add a new topic page:

1. Create a new folder, e.g. `app/networks/rnn`
2. Add a `page.tsx` file that uses the `ContentLayout` component:

```tsx
'use client';
import ContentLayout from '@/app/ui/ContentLayout';

export default function Page() {
  return (
    <ContentLayout
      title="Your Topic"
      intro={<p>This topic introduces...</p>}
      sections={[
        {
          title: 'Section Title',
          content: <p>Section content here...</p>,
        },
      ]}
    />
  );
}
```

See `app/languages/page.tsx` and `app/networks/page.tsx` for working examples.
