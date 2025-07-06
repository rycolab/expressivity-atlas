'use client';

interface Section {
  title: string;
  content: React.ReactNode;
}

interface ContentLayoutProps {
  title: string;
  intro?: React.ReactNode;
  sections: Section[];
}

export default function ContentLayout({ title, intro, sections }: ContentLayoutProps) {
  return (
    <main className="flex flex-col min-h-screen px-6 py-12 max-w-none mx-auto">
      {/* Page Title */}
      <header className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-gray-900 tracking-tight">{title}</h1>
      </header>

      {/* Introductory content (optional) */}
      {intro && (
        <div className="mb-12 text-gray-700 leading-relaxed text-lg text-center">
          {intro}
        </div>
      )}

      {/* Main Sections */}
      <div className="space-y-12">
        {sections.map((section, idx) => (
          <section key={idx}>
            <h2 className="text-2xl font-semibold text-gray-800 mb-3 text-center">{section.title}</h2>
            <div className="text-gray-700 leading-relaxed">
              {section.content}
            </div>
          </section>
        ))}
      </div>
    </main>
  );
}
