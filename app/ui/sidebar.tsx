'use client';

import {
  HomeIcon,
  DocumentIcon,
  FolderIcon,
  Bars3Icon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';
import React, { useState } from "react";

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
  const [mobileOpen, setMobileOpen] = useState(false);
  const [desktopOpen, setDesktopOpen] = useState(false);
  const pathname = usePathname();

  // For mobile: show sidebar with hamburger
  // For desktop: show sidebar only when hovered on hot zone or sidebar
  return (
    <>
      {/* Hamburger for mobile */}
      <button
        className="fixed top-4 left-4 z-50 md:hidden flex items-center p-2 rounded-lg bg-white shadow transition"
        onClick={() => setMobileOpen(true)}
        aria-label="Open sidebar"
      >
        <Bars3Icon className="w-7 h-7" />
      </button>

      {/* Overlay for mobile */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/40 md:hidden"
          onClick={() => setMobileOpen(false)}
          aria-label="Close sidebar overlay"
        />
      )}

      {/* Hot zone for desktop */}
      <div
        className="hidden md:block fixed top-0 left-0 h-screen w-3 z-50"
        onMouseEnter={() => setDesktopOpen(true)}
        onMouseLeave={() => setDesktopOpen(false)}
      />

      {/* Sidebar (fixed, not inside layout flex) */}
      <aside
        className={clsx(
          "fixed top-0 left-0 z-50 h-screen w-64 bg-white shadow-xl border-r border-gray-100 flex flex-col transition-transform duration-300",
          // Mobile
          mobileOpen ? "translate-x-0" : "-translate-x-full",
          "md:-translate-x-60", // Default desktop: hidden
          desktopOpen && "md:translate-x-0" // Show on desktop when hovered
        )}
        onMouseEnter={() => setDesktopOpen(true)}
        onMouseLeave={() => setDesktopOpen(false)}
        aria-label="Sidebar"
      >
        {/* Close Button for mobile */}
        <button
          className="md:hidden absolute top-4 right-4 z-50 p-1 rounded-lg bg-gray-50 hover:bg-gray-100"
          onClick={() => setMobileOpen(false)}
          aria-label="Close sidebar"
        >
          <XMarkIcon className="w-7 h-7" />
        </button>
        {/* Title & Logo */}
        <div className="p-6 font-extrabold text-xl text-gray-800 tracking-tight mb-4 select-none">
          Expressivity Atlas
        </div>
        {/* Navigation */}
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
                  onClick={() => setMobileOpen(false)}
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
                          onClick={() => setMobileOpen(false)}
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
        {/* User/Creator Info */}
        <div className="p-4 border-t border-gray-100 mt-auto">
          <a
            href="https://rycolab.io" 
            className="flex items-center gap-3 hover:bg-blue-50 p-2 rounded-lg transition"
            title="About Rycolab"
            target="_blank"
            rel="noopener noreferrer"
          >
            <img
              src="/lab.png"
              alt="Rycolab"
              className="w-10 h-10 rounded-full shadow"
            />
            <span className="font-semibold text-gray-800">Rycolab</span>
          </a>
        </div>
      </aside>
    </>
  );
}
