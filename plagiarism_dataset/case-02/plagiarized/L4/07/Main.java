import java.util.Scanner;
public class Main {
    public static void main(String[] args) {
        Scanner input = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder: ");
        double jejari = input.nextDouble();
        double panjang = input.nextDouble();
        System.out.println("The area is " + hitungArea(jejari));
        System.out.println("The volume of the cylinder is " + hitungVolume(panjang, jejari));
        }
    public static double hitungArea(double jejari){
        double area;
        return area = jejari * jejari * 3.14159;
    }
    public static double hitungVolume(double panjang, double jejari){
        double volume;
        return volume = hitungArea(jejari) * panjang;
    }
}
