'use client';

import {
  HomeIcon,
  DocumentIcon,
  FolderIcon,
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';
import React from "react";

const links = [
  { name: 'Home', href: '/', icon: HomeIcon },
  {
    name: 'Neural Networks',
    href: '/networks',
    subpages: [
      { name: 'Recurrent Neural Networks', href: '/networks/rnn' },
      { name: 'Transformers', href: '/networks/transformer' },
      { name: 'State-Space Models', href: '/networks/ssm' },
    ],
  },
  { name: 'Formal Languages', href: '/languages' },
  { name: 'Formal Logics', href: '/logics' },
  { name: 'Finite Automata', href: '/automata' },
  { name: 'Finite Monoids', href: '/monoids' },
  { name: 'Circuit Complexity', href: '/circuits' },
  { name: 'Programming Languages', href: '/programs' },
];

export default function SideBar() {
  const pathname = usePathname();
  return (
    <aside className="w-64 bg-white h-screen flex flex-col shadow-xl border-r border-gray-100">
      <div className="p-6 font-extrabold text-xl text-gray-800 tracking-tight mb-4 select-none">
        Expressivity Atlas
      </div>
      <ul className="flex-1 px-2 space-y-2">
        {links.map((link) => {
          const Icon = link.icon ?? FolderIcon;
          const isActive = pathname === link.href || pathname.startsWith(link.href + '/');
          return (
            <li key={link.name}>
              <Link
                href={link.href}
                className={clsx(
                  "flex items-center gap-3 rounded-xl px-4 py-3 text-base font-medium transition-colors duration-200",
                  isActive
                    ? "bg-blue-50 text-blue-700 shadow"
                    : "hover:bg-blue-50 text-gray-700"
                )}
              >
                <span className={clsx(
                  "flex items-center justify-center rounded-lg p-1 transition-all",
                  isActive
                    ? "bg-blue-100"
                    : "bg-gray-100"
                )}>
                  <Icon className="w-6 h-6" />
                </span>
                <span>{link.name}</span>
              </Link>
              {/* Render subpages if present and parent link is active */}
              {isActive && link.subpages && (
                <ul className="ml-7 mt-1 border-l-2 border-blue-100 pl-3 space-y-1">
                  {link.subpages.map((subpage) => (
                    <li key={subpage.name}>
                      <Link
                        href={subpage.href}
                        className={clsx(
                          "flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                          pathname === subpage.href
                            ? "bg-blue-100 text-blue-700 font-semibold shadow"
                            : "hover:bg-blue-50 text-gray-600"
                        )}
                      >
                        <DocumentIcon className="w-4 h-4" />
                        <span>{subpage.name}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          );
        })}
      </ul>
      <div className="p-4 border-t border-gray-100 flex items-center gap-3 mt-auto">
        <img
          src="/lab.png" // <- use your icon file here
          alt="Rycolab"
          className="w-10 h-10 rounded-full shadow"
        />
        <div>
          <div className="font-semibold text-gray-800">Rycolab</div>
        </div>
        {/* Optional: Add settings/logout button here */}
      </div>
    </aside>
  );
}
