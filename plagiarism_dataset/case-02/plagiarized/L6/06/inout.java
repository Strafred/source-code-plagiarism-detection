import java.util.Scanner;
public class inout{
    public static void main(String[] args) {
        double rad,l;
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder : ");
        rad = sc.nextDouble();
        l = sc.nextDouble();
        hitungVol(hitungAr(rad), l);
    }
    public static void hitungVol(double ar,double l)
    {
        double vol= ar*l;
        System.out.println("The volume of the cylinder is " + vol);
    }
    public static double hitungAr(double rad)
    {
        double ar= rad*rad*3.14159;
        System.out.println("The area is " + ar);
        return ar;
    }
}