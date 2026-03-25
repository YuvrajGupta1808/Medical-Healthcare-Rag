"use client";
import { cn } from "@/lib/utils";
import React, { ReactNode } from "react";

interface AuroraBackgroundProps extends React.HTMLAttributes<HTMLDivElement> {
  children: ReactNode;
  showRadialGradient?: boolean;
}

export const AuroraBackground = ({
  className,
  children,
  showRadialGradient = true,
  ...props
}: AuroraBackgroundProps) => {
  return (
    <main>
      <div
        className={cn(
          "relative flex flex-col  h-[100vh] items-center justify-center bg-zinc-950  text-slate-950 transition-bg",
          className
        )}
        {...props}
      >
        <div className="absolute inset-0 overflow-hidden">
          <div
            //   backdrop-filter: blur(10px);
            className={cn(
              `
            [--white-gradient:linear-gradient(to_bottom,white,white_35%,rgba(255,255,255,0))]
            [--dark-gradient:linear-gradient(to_bottom,black,black_35%,rgba(0,0,0,0))]
            [--aurora:repeating-linear-gradient(100deg,var(--blue-500)_10%,var(--emerald-500)_15%,var(--indigo-500)_20%,var(--violet-500)_25%,var(--blue-500)_30%)]
            [background-image:var(--dark-gradient),var(--aurora)]
            [background-size:300%,_200%]
            [background-position:50%_50%,_50%_50%]
            filter blur-[10px] invert-0
            after:content-[""] after:absolute after:inset-0 after:[background-image:var(--dark-gradient),var(--aurora)]
            after:[background-size:200%,_100%] 
            after:animate-aurora after:[background-attachment:fixed]
            pointer-events-none
            absolute -inset-[10px] opacity-50 will-change-transform`,
              showRadialGradient &&
                `[mask-image:radial-gradient(ellipse_at_100%_0%,black_10%,transparent_70%)]`
            )}
          ></div>
        </div>
        {children}
      </div>
    </main>
  );
};
