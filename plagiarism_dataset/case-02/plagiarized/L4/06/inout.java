import java.util.Scanner;
public class inout{
    public static void main(String[] args) {
        double rad,l,ar,vol;
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder : ");
        rad = sc.nextDouble();
        l = sc.nextDouble();
        ar = hitungAr(rad);
        vol = hitungVol(ar,l);
        System.out.println("The area is " + ar);
        System.out.println("The volume of the cylinder is " + vol);
    }
    public static double hitungVol(double ar,double l)
    {
        return ar*l;
    }
    public static double hitungAr(double rad)
    {
        return rad*rad*3.14159;
    }
}