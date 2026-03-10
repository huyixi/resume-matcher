import { readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { BODY_FONT_MAP, HEADER_FONT_MAP, settingsToCssVars } from '@/lib/types/template-settings';

describe('template font configuration', () => {
  it('routes resume font selections through the global font tokens', () => {
    expect(HEADER_FONT_MAP.serif).toBe('var(--font-serif)');
    expect(HEADER_FONT_MAP['sans-serif']).toBe('var(--font-sans)');
    expect(HEADER_FONT_MAP.mono).toBe('var(--font-mono)');
    expect(BODY_FONT_MAP.serif).toBe('var(--font-serif)');
    expect(BODY_FONT_MAP['sans-serif']).toBe('var(--font-sans)');
    expect(BODY_FONT_MAP.mono).toBe('var(--font-mono)');
  });

  it('keeps the default generated CSS variables on the shared font tokens', () => {
    const cssVars = settingsToCssVars();

    expect(cssVars['--header-font']).toBe('var(--font-serif)');
    expect(cssVars['--body-font']).toBe('var(--font-sans)');
  });

  it('includes Chinese-capable fallback fonts in the global font stacks', () => {
    const cssPath = path.resolve(process.cwd(), 'app/(default)/css/globals.css');
    const css = readFileSync(cssPath, 'utf8');

    expect(css).toContain('Noto Sans CJK SC');
    expect(css).toContain('Noto Serif CJK SC');
    expect(css).toContain('Microsoft YaHei');
  });
});
